# Running tests on native RBE (Cloud TPU v5e)

This is the how-to. For how the cluster is built, see
[native_rbe_architecture.md](native_rbe_architecture.md).

The v5e presubmit runs on the shared 8-chip `ct5lp` runner, at most 8 tests at a
time, and a full run takes 44–79 minutes. The `rbe-tpu-oss` Buildbarn cluster
spreads the single-chip tests across 32 Cloud TPU v5e workers (one chip each)
and a pool of CPU workers for compilation.

You can use it three ways:

1.  [On a pull request](#1-on-a-pull-request)
2.  [From a GCE VM in the `rbe-tpu-oss` VPC](#2-from-a-gce-vm-in-the-vpc)
3.  [From a cloudtop](#3-from-a-cloudtop)

## 1. On a pull request

Add this label to the PR:

*   `run-rbe`: runs the RBE job alongside the normal presubmit.

You can also start it by hand with `workflow_dispatch`.

The `Presubmit on RBE` job writes the repository secrets `RBE_CA_CERT`,
`RBE_CLIENT_CERT`, and `RBE_CLIENT_KEY` to `/tmp/rbe-certs/`, then runs `bazel
test --config=ci_rbe_presubmit` against the Private Service Connect endpoint
`grpcs://10.202.0.67:8980`. The job skips PRs from forks, which keeps fork code
off the workers. GitHub doesn't give fork PRs the repository secrets either.

## 2. From a GCE VM in the VPC

VMs in the `rbe-tpu-oss` VPC reach the frontend through the internal load
balancer at `grpcs://10.202.0.15:8980`.

### Create the VM

```bash
gcloud compute instances create <name> --project=rbe-tpu-oss --zone=us-east5-a \
  --machine-type=n2-standard-16 --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=100GB --boot-disk-type=pd-ssd --scopes=cloud-platform
```

The VM runs as the project's default compute service account, which can already
read the three `rbe-*` secrets in Secret Manager (see
[Credentials](#credentials)).

### Install what stock Debian 12 is missing

```bash
sudo apt-get update -qq
sudo apt-get install -y -qq git build-essential curl wget

sudo curl -fsSLo /usr/local/bin/bazel \
  https://github.com/bazelbuild/bazelisk/releases/download/v1.25.0/bazelisk-linux-amd64
sudo chmod +x /usr/local/bin/bazel

sudo mkdir -p -m 755 /etc/apt/keyrings
wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg \
  | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null
sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
  | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt-get update -qq && sudo apt-get install -y -qq gh
```

### Clone the repo

```bash
gh auth login
gh auth setup-git
git clone https://github.com/google-pytorch/torch_tpu.git
cd torch_tpu
```

### Get credentials and run

Fetch the certs as described in [Credentials](#credentials), then:

```bash
bazel test --config=rbe_native_tpu --remote_executor=grpcs://10.202.0.15:8980 \
  //tests:autoload_test_tpu
```

Use `10.202.0.15`. The frontend's certificate covers `10.202.0.15` (VPC internal
load balancer), `10.202.0.67` (CI Private Service Connect endpoint), `127.0.0.1`
and `localhost` (port-forwarding), and two leftovers from earlier setups,
`10.202.0.3` and `rbe.torchtpu.internal`. It doesn't cover the GKE node IP
`10.202.0.16`. Any other address fails TLS hostname validation with `General
OpenSslEngine problem`. Adding an endpoint means reissuing the cert.

## 3. From a cloudtop

A cloudtop is outside the VPC, so you forward the frontend port to it:

```bash
gcloud container clusters get-credentials rbe-tpu-oss-east5-cluster \
  --region=us-east5 --project=rbe-tpu-oss
kubectl port-forward -n buildbarn svc/frontend 8980:8980
```

Leave that running, fetch the certs as described in [Credentials](#credentials),
and in another shell:

```bash
bazel test --config=rbe_native_tpu //tests:autoload_test_tpu
```

`rbe_native_tpu` points at `grpcs://127.0.0.1:8980` by default, which is the
port forward.

## Credentials

Reading the three secrets needs `roles/secretmanager.secretAccessor` on them.
VMs created as in [section 2](#2-from-a-gce-vm-in-the-vpc) already have it
through the default compute service account.

`.bazelrc` reads the certs from `/tmp/rbe-certs/`:

```bash
umask 077
mkdir -p /tmp/rbe-certs
gcloud secrets versions access latest --secret=rbe-ca-cert --project=rbe-tpu-oss > /tmp/rbe-certs/rbe_ca.crt
gcloud secrets versions access latest --secret=rbe-client-crt --project=rbe-tpu-oss > /tmp/rbe-certs/rbe_client.crt
gcloud secrets versions access latest --secret=rbe-client-key --project=rbe-tpu-oss > /tmp/rbe-certs/rbe_client.key
```

On a shared machine, another user can create `/tmp/rbe-certs` before you do.
Write the certs under your home directory instead, then point Bazel at them from
`user.bazelrc` in the repo root (gitignored and loaded last). Use absolute
paths; Bazel doesn't expand `~`. Run this from the repo root:

```bash
umask 077
d="$HOME/.config/torch_tpu/certs"
mkdir -p "$d"
gcloud secrets versions access latest --secret=rbe-ca-cert --project=rbe-tpu-oss > "$d/rbe_ca.crt"
gcloud secrets versions access latest --secret=rbe-client-crt --project=rbe-tpu-oss > "$d/rbe_client.crt"
gcloud secrets versions access latest --secret=rbe-client-key --project=rbe-tpu-oss > "$d/rbe_client.key"
cat >> user.bazelrc <<EOF
common:rbe_native_tpu --tls_certificate=$d/rbe_ca.crt
common:rbe_native_tpu --tls_client_certificate=$d/rbe_client.crt
common:rbe_native_tpu --tls_client_key=$d/rbe_client.key
EOF
```

Never commit these files. `bazel/certs/` and `*.key` are in `.gitignore` as a
safety net.

## Running any test

Any Bazel test target works with `--config=rbe_native_tpu`, with the usual flags
(`--test_filter`, `--runs_per_test`, `--test_output=streamed`):

```bash
bazel test --config=rbe_native_tpu //tests/pallas:pallas_test
bazel test --config=rbe_native_tpu //tests/pallas:pallas_test --runs_per_test=20
bazel test --config=rbe_native_tpu //tests:ops_test_tpu-v5lite
```

Compilation runs on the CPU workers and the test itself runs on a TPU worker.

Each TPU worker has one v5e chip. Targets tagged `requires-tpu-v5lite:8` need an
8-chip host and can't run here.

## Measured times

<!-- mdformat off(GitHub renders this file; keep the single-line table) -->
What                                                 | Time
:--------------------------------------------------- | :-------------------------------------------
`//tests:autoload_test_tpu` on a v5e worker          | ~12s test time
`//csrc/common:flags_test`, build and test from a VM | 8s total
`//tests:ops_test_tpu-v5lite`, 50 shards             | 3m 38s on RBE vs 4m 48s on one 8-chip TPU VM
<!-- mdformat on -->
