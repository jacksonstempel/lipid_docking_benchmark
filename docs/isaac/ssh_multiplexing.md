# ISAAC SSH multiplexing (8-hour “no re-login” workflow)

Goal: avoid typing a password and doing Duo/MFA every time you run `rsync`, `scp`, or `ssh`.

## What this is

SSH “multiplexing” lets multiple SSH sessions reuse one authenticated connection. You log in (and Duo) once, and then transfers/commands reuse that session for a set time window.

This is controlled by:

- `ControlMaster auto`: allow a shared master connection
- `ControlPath ...`: where the shared socket lives on your laptop
- `ControlPersist 8h`: keep the master connection alive for ~8 hours after the last use

## One-time setup (on your laptop)

Edit `~/.ssh/config` and add something like:

```sshconfig
Host isaac
  HostName login2.isaac.utk.edu
  User jstempel
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 8h
  ServerAliveInterval 60
  ServerAliveCountMax 3
```

Make sure permissions are sane:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/config
```

## How you use it day-to-day

1) Start the master session (you’ll do Duo once):

```bash
ssh isaac
```

You can exit immediately; the connection stays reusable:

```bash
exit
```

2) Run commands/transfers without repeated Duo:

```bash
ssh isaac "hostname; date"
```

```bash
rsync -av --progress isaac:"$SCRATCHDIR/gnina/runs/<run_name>/flat/" ./flat/
```

## Check / close the master connection

Check:

```bash
ssh -O check isaac
```

Close:

```bash
ssh -O exit isaac
```

## Common issues

- If `rsync` still prompts for Duo every time:
  - The master session may not be running; run `ssh isaac` once to start it.
  - The `ControlPath` directory must exist and be writable (`~/.ssh`).
- If you see “Control socket … Operation not permitted”:
  - The socket path may not be writable in your environment, or you may be running in a restricted sandbox. Use your normal terminal on your laptop.

