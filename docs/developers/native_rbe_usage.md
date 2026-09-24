# Running tests on native RBE (Cloud TPU v5e)

This is the how-to. For how the cluster is built, see
[native_rbe_architecture.md](native_rbe_architecture.md).

The single-chip v5e presubmit runs one test at a time on the shared `ct5lp`
runner and takes 54–83 minutes. The `rbe-tpu-oss` Buildbarn cluster spreads the
same tests across about 30 single-chip Cloud TPU v5e workers and a pool of CPU
workers for compilation.

You can use it three ways:

1.  [On a pull request](#1-on-a-pull-request)
2.  [From a GCE VM in the `rbe-tpu-oss` VPC](#2-from-a-gce-vm-in-the-vpc)
3.  [From a cloudtop](#3-from-a-cloudtop)

## 1. On a pull request

Add one of these labels to the PR:

*   `ci:replace-tpu-v5`: skips the serial v5e runner and sends the v5e tests to
    RBE.
*   `run-rbe`: runs the RBE job alongside the normal presubmit.

You can also start it by hand with `workflow_dispatch`.

The `Presubmit on RBE` job writes the repository secrets `RBE_CA_CERT`,
`RBE_CLIENT_CERT`, and `RBE_CLIENT_KEY` to `/tmp/rbe-certs/`, then runs `bazel
test --config=ci_rbe_presubmit` against the Private Service Connect endpoint
`grpcs://10.202.0.67:8980`. The job doesn't run for PRs from forks, because
forks don't get repository secrets.

## 2. From a GCE VM in the VPC

VMs in the `rbe-tpu-oss` VPC reach the frontend through the internal load
balancer at `grpcs://10.202.0.15:8980`.

### Create the VM

```bash
gcloud compute instances create <name> --project=rbe-tpu-oss --zone=us-east5-a \
  --machine-type=n2-standard-16 --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=100GB --boot-disk-type=pd-ssd --scopes=cloud-platform
```

The VM's service account needs read access to the three `rbe-*` secrets in
Secret Manager (see [Credentials](#credentials)).

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

Use `10.202.0.15`. The frontend's certificate covers that address, and other
internal addresses fail the TLS hostname check with `General OpenSslEngine
problem`.

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

Bazel reads the client certificate from `/tmp/rbe-certs/`:

```bash
mkdir -p /tmp/rbe-certs
gcloud secrets versions access latest --secret=rbe-ca-cert --project=rbe-tpu-oss > /tmp/rbe-certs/rbe_ca.crt
gcloud secrets versions access latest --secret=rbe-client-crt --project=rbe-tpu-oss > /tmp/rbe-certs/rbe_client.crt
gcloud secrets versions access latest --secret=rbe-client-key --project=rbe-tpu-oss > /tmp/rbe-certs/rbe_client.key
chmod 600 /tmp/rbe-certs/rbe_client.key
```

Never commit these files. `bazel/certs/` is in `.gitignore` for that reason.

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

| What                                 | Time                                  |
| :----------------------------------- | :------------------------------------ |
| `//tests:autoload_test_tpu` on a v5e | ~12s test time                        |
: worker                               :                                       :
| `//csrc/common:flags_test`, build    | 8s total                              |
: and test from a VM                   :                                       :
| `//tests:ops_test_tpu-v5lite`, 50    | 3m 38s on RBE vs 4m 48s on one 8-chip |
: shards                               : TPU VM                                :
