# How the TPU relay works, in plain English

No jargon, no flags. If you want the wiring diagram, read
[RELAY_ARCHITECTURE.md](RELAY_ARCHITECTURE.md).

## The problem

Before every change lands, we run 57 tests that need a real TPU chip. That used
to take about an hour on one machine.

Here is the thing though: almost none of that hour is testing. It is
**compiling** — turning a few million lines of C++ into something runnable. The
actual testing is a few minutes at the end.

So we were tying up a rare, expensive TPU machine for an hour to do 50 minutes
of work that any laptop could have done, and 7 minutes of work that only a TPU
can do.

## The fix

Split the job in two and give each half to hardware that suits it.

**Compiling** goes to a build farm — a few hundred ordinary machines that spin
up, compile their share, hand back the result, and disappear. They have never
heard of a TPU and do not need to.

**Testing** goes to a pool of 28 TPU machines we keep running all the time,
spread across four data centers.

Your machine sits in the middle and runs the show. It sends the compile work
out, collects everything that comes back, then hands finished tests to whichever
TPU machine is free.

An hour becomes about seven minutes.

## Walking through a run

Say you kick off the presubmit.

**1. Your machine works out what to test.** 57 targets.

**2. It ships the compiling to the build farm.** Two hundred machines chew on it
at once. Anything already built by someone else recently gets pulled from a
shared cache instead of rebuilt.

**3. Everything comes back to your machine.** This is the part people find
surprising: the build farm does *not* talk to the TPUs. It cannot. Everything
funnels back through you and then out again.

**4. Your machine borrows TPUs.** Not creates — borrows. The 28 machines are
already running. Your run puts its name on them, uses them, and takes its name
off when it is finished. Nobody's hardware gets created or destroyed by a test
run, which means a crashed run cannot delete a machine somebody else is using.

**5. Tests go out 28 at a time.** Each test grabs a free TPU machine, copies
over what it needs, runs there, and sends back the result. Then the next test
takes that machine.

**6. You get a verdict.** Pass or fail per target, same as always.

## Why it is fast

Three reasons, roughly in order of how much they matter.

### The TPU machines are already on

Starting a TPU machine from scratch takes minutes. Twenty-eight of them takes a
while. So we do not. They stay up, and runs take turns.

This is also why the script can only *borrow* them. Letting a test run delete
hardware is how you end up deleting hardware somebody else is mid-way through
using.

### We stopped sending the same 500 MB over and over

Every test needs the same big pile of shared stuff — Python, PyTorch, the TPU
libraries, our own compiled code. That is a couple of gigabytes, and it is
identical for all 57 tests.

So it gets sent to each TPU machine **once** and left there. Each test then only
sends its own handful of files.

One file alone, a compiled library, used to be 493 MB of a 499 MB transfer, sent
again for every single test. Moving it into the shared pile is most of the
speedup right there: one test's transfer went from 499 MB to under 200 KB, and
the time spent packing it from about 15 seconds to a fifth of a second. What
each test sends now is a few hundred kilobytes, not half a gigabyte.

What is left is getting that shared pile out in the first place. It is about a
gigabyte per machine once packed, and all 28 machines are fed from your one
machine's connection — so roughly 30 GB after a change that rebuilds everything,
and about a third of that on a normal change where only our own compiled code
moves. That, not what each test sends, is the part that will strain first if the
pool gets bigger.

### Big tests get split, and the pieces share

A large test gets chopped into 30-50 pieces that run in parallel. All those
pieces need exactly the same files. So we unpack the files once and let the
pieces share one copy. That turns 20 seconds of unpacking into under a second,
per piece.

## Things that go wrong, and what happens

**The TPU machine disappears.** Our fleet is on-demand, so this should not
happen on its own — nobody can take these machines away mid-run. But the tool
handles it anyway, because the same scripts can create cheaper "spot" machines
that Google Cloud *can* reclaim without warning. If a machine does vanish, the
test says so plainly rather than blaming your code.

We were careful to separate "the machine is gone" from "I could not reach the
machine to ask." Early on those got lumped together and a run blamed Google
Cloud for 64 failures that were actually a local problem on our side.

**The machine restarts.** It comes back looking healthy but with its temporary
files wiped, including that big shared pile. The test would run without a Python
interpreter and die with a useless error. So we check before sending anything,
and pull that machine out of rotation if the pile is missing.

**Two people run at once.** The second person waits in line rather than
splitting the machines. A run is seven minutes — waiting your turn for all 28
chips beats both of you limping along with 14 each.

**Something goes wrong with the plumbing itself.** Every failure path still
writes a proper test report. An infrastructure problem shows up as a named
failure with an explanation, not as an empty result you have to go digging into.

## What you can run on it

It started as a presubmit-only tool, tied to pull requests. It is not any more.

You can run any bazel test on the fleet:

```
scripts/relay_bazel_test.sh //tests/pallas:pallas_test
scripts/relay_bazel_test.sh //tests/pallas:pallas_test -- --runs_per_test=20
scripts/relay_bazel_test.sh //tests/...
```

It borrows the fleet, runs, and gives the machines back however it exits,
including if you hit Ctrl-C.

You also no longer need to be on a corp workstation. A cloud VM works, and it
actually has a faster connection to the chips. It figures out which kind of
machine it is on by itself.

One real limit, and it is the hardware rather than the tool: each test gets one
chip. A test that needs eight chips at once cannot pass here no matter how you
run it.

## The short version

Compiling is parallel and needs no special hardware, so it goes to a few hundred
cheap machines. Testing needs a TPU and barely any CPU, so it goes to a small
pool of TPU machines that stay switched on. Your machine is the middleman. Stop
re-sending the parts that never change.

Roughly an hour, down to roughly seven minutes.
