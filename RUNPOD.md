# Running this on RunPod

Use a **GPU Pod**, not Serverless. Serverless workers have no shell, no
`SYS_ADMIN` (so no Nsight counters), and worker recycling means two sweeps
might not have run on the same physical card.

## One-time, at home

Add an SSH key to RunPod before you ever start a pod, otherwise you're stuck
in the browser terminal.

```bash
ssh-keygen -t ed25519 -C "runpod"          # if you don't have one
cat ~/.ssh/id_ed25519.pub
```

Paste that into RunPod → Settings → SSH Public Keys.

Push the repo to GitHub. That's how the code gets onto the pod; there's no
good reason to drag files through a browser.

```bash
git init
git add .
git commit -m "kernel bench harness"
gh repo create gpu-kernel-bench --private --source=. --push
```

## Starting a pod

1. RunPod → Pods → Deploy.
2. Pick **RTX 4090**. Community Cloud is cheaper, Secure Cloud is more
   reliable. Either is fine.
3. Template: **RunPod PyTorch** (any 2.x image). It already has torch,
   triton and a matching CUDA toolkit, which is what makes `cpp_extension`
   work without setup.
4. Container disk 20 GB is plenty. Volume disk 20 GB, mounted at
   `/workspace`. Only `/workspace` survives a stop/start.
5. Deploy, wait for it to go green, then Connect → copy the SSH command.

```bash
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519
```

## Setting up on the pod

```bash
cd /workspace                       # anything outside this is lost on restart
git clone https://github.com/<you>/gpu-kernel-bench.git
cd gpu-kernel-bench

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# reuse the image's torch instead of downloading another 2.5 GB
uv venv --system-site-packages
uv pip install matplotlib numpy
```

## First thing to check

```bash
uv run kb env
```

Look at `profile_tier`.

* **A**: counters work, everything in the README applies.
* **B**: timing works, Nsight doesn't. If `ncu not on PATH`, try installing
  it (below). If it says `ERR_NVGPUCTRPERM`, the container wasn't given
  `SYS_ADMIN` and no flag fixes it from inside. Try a different template or
  Secure Cloud; if neither works, you're on tier B on RunPod and Nsight work
  has to happen on your own machine.

If `ncu` is missing but the pod has the CUDA toolkit:

```bash
ls /usr/local/cuda/bin/ncu                      # often already there
export PATH=/usr/local/cuda/bin:$PATH
```

## Actually running

```bash
uv run kb bench --op vector_add --dtype fp16    # harness sanity check first
uv run kb bench --op all --dtype fp16
uv run kb profile --op rmsnorm --impl cuda --config-index 2
```

`vector_add` is the self-test. All four implementations should land within a
few percent of each other and near the measured copy roof. If they don't,
stop and fix the harness.

Long sweeps under `tmux` so an SSH drop doesn't kill them:

```bash
tmux new -s bench
# ... run things ...
# ctrl-b then d to detach, tmux attach -t bench to come back
```

## Getting results back

Results are small JSON and PNG, so commit them.

`.gitignore` excludes `results/*.json` and `figures/*` by default, since I
don't want every experiment in git. Force-add the ones worth keeping:

```bash
git add -f results/rmsnorm_fp16_NVIDIA_GeForce_RTX_4090_*.json
git add -f figures/rmsnorm_fp16_*
git commit -m "rmsnorm fp16 sweep, 4090"
git push
```

Pushing from the pod needs credentials. Simplest is a fine-grained GitHub PAT
with contents:write on this repo only:

```bash
git remote set-url origin https://<token>@github.com/<you>/gpu-kernel-bench.git
```

That writes the token into `.git/config` on the pod, so use a scoped token
and revoke it when you're done with the pod.

Or pull it down instead:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 \
    'root@<ip>:/workspace/gpu-kernel-bench/results/*.json' ./results/
```

## Before you close the tab

**Terminate the pod, don't just stop it.** A stopped pod still bills for the
volume. Anything you want to keep is already in git.
