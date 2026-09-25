# Loading data at scale — a guided lesson

The [data layer spec](data.md) says *what* to build. This lesson explains *why*
the disk-backed path looks the way it does, with experiments you can run on
this machine. By the end you should be able to reason about any data-loading
design — dsio's or someone else's — and know which knob to turn when it is
slow.

**How to run the experiments**: everything is in the workspace
(`forge workspace build` builds `fp` → `dsio`). The benchmarks live in dsio:

```bash
cd ../dsio
scripts/bench.sh                       # the dsio matrix
scripts/bench.sh --fio                 # + the fio ceiling (needs FIO=...)
scripts/bench.sh --large               # + the 24 GiB cache-busting pass
forge bench --benchmark_filter=BM_DatasetStream
```

Numbers quoted here are from this machine (WSL2, ext4 on a vhdx, O_DIRECT) and
are recorded in [`dsio/docs/roadmap.md`](../../dsio/docs/roadmap.md). They are
data points, not constants — the point is the *shape* of the curves.

---

## Lesson 1 — Two costs: bandwidth and latency

Storage has two numbers, and they differ by orders of magnitude:

| Path | Bandwidth | Latency |
|---|---|---|
| Guest RAM (page cache, mmap) | ~300 GiB/s | ~100 ns |
| NVMe through this WSL2 vhdx | 2–6 GiB/s | ~100 µs per request |
| A 4 KiB read, done one at a time | ~35 MiB/s | 100 µs each |

That last row is the trap: a fast device used one request at a time looks
slow. 4 KiB / 100 µs = 40 MB/s, and that is exactly what the benchmark
measures (`BM_Sequential/4096` ≈ 35 MiB/s).

**Try it.** In dsio:

```bash
forge bench --benchmark_filter='BM_Sequential'
```

| Block size | Throughput |
|---|---|
| 4 KiB | ~35 MiB/s |
| 64 KiB | ~350 MiB/s |
| 1 MiB | ~1.8 GiB/s |

Same device, same bytes — only the request size changed.

**Checkpoint.** You can explain why "the disk is fast" and "this loop is slow"
are both true, and why request size matters before anything else.

---

## Lesson 2 — The page cache is not free, and O_DIRECT is not magic

Normally a read copies data twice: device → page cache (kernel RAM) → your
buffer. The page cache makes repeated reads of the same bytes fast (RAM
speed), but it costs a copy and it holds memory — memory your model wants.

`O_DIRECT` asks the kernel to DMA straight into your buffer. No copy, no cache
pollution. The price is a contract:

- the buffer address, the file offset and the length must all be multiples of
  the device's block size (4096 here);
- a filesystem without O_DIRECT support (tmpfs!) refuses to open the file.

dsio encodes the contract in two helpers:

```cpp
auto block = dsio::block_size("tokens.bin");   // 4096 here
auto buffer = fp::AlignedBuffer::alloc(1 << 20, block.value());
// buffer.value().data() is block-aligned; the size is rounded up
```

**Try it.** Ask for a misaligned read and watch the contract bite:

```cpp
auto file = dsio::File::open("tokens.bin", dsio::OpenMode::DirectRead);
auto read = file.value().pread(/*offset=*/1, buffer.value().span());
// read.error().code == EINVAL — a value, not a crash
```

`dsio::align_up(1, 4096)` is the fix. Over-aligning is always safe: 4096 is
the conservative value (`block_size` returns the device's logical block size
when it can, the filesystem's otherwise).

**Checkpoint.** You can say what O_DIRECT buys, what it costs, and why
alignment is measured in *blocks*, not bytes. You can name a filesystem where
it does not work at all (`/tmp` on tmpfs).

---

## Lesson 3 — Queue depth: how to turn latency into bandwidth

If one request takes 100 µs, issue many at once. The device serves them in
parallel and the effective throughput multiplies — until the device's own
queue saturates.

dsio's `Backend` is that idea: submit a batch of reads, reap completions as
they finish (out of order, each carrying the token you attached).

```cpp
auto backend = dsio::open_backend(file.fd(), {.queue_depth = 16});
backend.value()->submit(requests);            // many at once
backend.value()->reap(completions, /*min=*/1); // as they complete
```

**Try it.**

```bash
forge bench --benchmark_filter='BM_Queued'
```

| Depth | io_uring | thread pool |
|---|---|---|
| 1 | ~1.8 GiB/s | ~1.6 GiB/s |
| 4 | ~2.3 GiB/s | ~2.4 GiB/s |
| 16 | ~2.4 GiB/s | ~2.7 GiB/s |
| 64 | ~2.9 GiB/s | ~2.7 GiB/s |

Two lessons in one table: depth turns latency into bandwidth, and the
*fallback* (thread pool + `pread`) reaches the same ballpark when io_uring is
unavailable — which is why dsio always keeps it.

**Checkpoint.** You can explain queue depth, why completions are reaped out of
order, and why every request needs its own buffer.

---

## Lesson 4 — Datasets are shards; the manifest is the contract

One 200 GB file is awkward: no parallel access, no resumability, no shuffle
granularity. A **shard set** is a list of byte ranges — usually one file per
shard — described by a manifest:

```text
# dsio shard manifest v1
tokens-000.bin	0	1073741824
tokens-001.bin	0	1073741824
```

`dsio::ShardSet::load` parses it (relative paths resolve against the
manifest's directory; offsets must be 4096-aligned), `discover` builds one
shard per file in a directory, and `save` writes it back.

**Why shard size is the single biggest knob on this machine.** The vhdx read
path ramps with the length of an *uninterrupted sequential run* (measured with
fio, reproduced end-to-end with dsio):

| Contiguous run | Throughput |
|---|---|
| 32 MiB | ~2.1 GiB/s |
| 128 MiB | ~4.3 GiB/s |
| 512 MiB | ~5.6 GiB/s |
| 1 GiB | ~6.0 GiB/s |
| 4 GiB | ~6.2 GiB/s |

dsio end-to-end: 8 × 32 MiB shards → **2.75 GiB/s**, 4 × 1 GiB shards →
**6.25 GiB/s**. Same code, same bytes: the layout is 2.3×.

**Try it.**

```bash
DSIO_BENCH_SHARDS=build/bench-data/shards-1g scripts/bench.sh   # the 1 GiB layout
scripts/bench.sh --large                                        # 4 GiB windows
```

**Checkpoint.** You can choose a shard size with a reason (throughput ramp vs
shuffle granularity), and you know that shuffle *above* dsio is how you keep
both.

---

## Lesson 5 — Streaming with bounded memory

Training reads epochs, not files. The pipeline must hand out batches while
keeping memory flat:

- a worker reads **ahead** of the consumer (`prefetch` batches);
- reads within a shard stay in flight (`read_depth` segments);
- batch buffers are **reused**, which is why the callback's span is only valid
  for the duration of the call.

```cpp
dsio::BatchOptions options;
options.batch_bytes = 4 << 20;   // per callback
options.prefetch    = 2;         // batches ahead of the consumer
options.read_depth  = 16;        // segment reads in flight per shard

dsio::for_each_batch(shards, options, [](std::span<const std::byte> batch) {
  consume(batch);                // copy what you keep!
  return true;                   // false stops early
});
```

Resident memory is bounded and predictable:
`(prefetch + 2) × batch_bytes + read_depth × segment`.

**Try it.** Watch the knobs move the needle:

```bash
forge bench --benchmark_filter='BM_DatasetReadDepth|BM_DatasetMatrix'
```

Depth 16 with the 256 KiB default segment (≈ 4 MiB in flight) is the sweet
spot on this machine; depth 1 drops to ~0.8 GiB/s, depth 64 gains nothing.
And note `BM_TokenPack` (copying each batch into `u16` storage) runs at the
same speed as streaming alone — the copy is free at these bandwidths, so
*clarity wins*.

**Checkpoint.** You can size `batch_bytes`/`prefetch`/`read_depth` for a
memory budget, and you can explain why the callback span is ephemeral.

---

## Lesson 6 — Measuring like an engineer

I/O benchmarks lie easily. The rules that saved us here:

- **Report real time, not CPU time.** Google Benchmark's
  `bytes_per_second` uses CPU time unless `UseRealTime()` is set; I/O waits
  are not CPU time, so the CPU-time number was ~10× too high until we fixed
  it.
- **Beware the host cache.** A freshly written file reads faster than a cold
  one (we saw 5.1 → 3.0 GiB/s on the same file). Evict (read a much larger
  file) before trusting a number.
- **Use an independent reference.** `fio` (`scripts/bench.sh --fio`) is the
  ground truth; dsio matches or beats it on every layout, which is the actual
  claim worth making.
- **Repeat and note the machine state.** The same benchmark moved 2–4% with
  load average 4.59 vs 0.03.
- **Track regressions.** `scripts/bench.sh --save`, then `--compare` after
  every change.

**Try it.**

```bash
scripts/bench.sh --save          # baseline
# ... change a knob ...
scripts/bench.sh --compare       # did it help?
```

**Checkpoint.** You can design a measurement that would convince a skeptic —
including what would falsify it.

---

## Lesson 7 — Crossing the bus: CPU and GPU data paths

The GPU cannot read your pointers. Bytes must land in *device* memory, and
there are two ways:

1. **Host staging** — dsio fills host batches, ml copies them into device
   buffers (`fp::gpu::Buffer`, pinned `HostBuffer`). Universal: works on
   WSL2, on the CPU fallback, on every GPU. Costs one copy over the bus.
2. **GPUDirect Storage (cuFile)** — the NVMe DMAs straight into device memory:
   `dsio::open_device_sink()` + `dsio::read_into()`, no host bounce buffer.
   Needs a `DSIO_WITH_CUFILE` build, an NVIDIA GPU and `nvidia-fs`; not
   available on WSL2.

dsio's seam is `Sink`: a region, what kind of memory it is, and a `commit`
after the bytes land. The rule is the same as everywhere in this stack — the
CPU path is always available and always tested; the accelerated path is
opt-in and reports why it is unavailable:

```cpp
if (!dsio::gpu_direct_available())
  // fall back to host staging; open_device_sink() says what is missing
```

And the split of concerns: **`fp::gpu` computes on the device; dsio's sink is
how the bytes get there.**

**Checkpoint.** You can draw the two paths with their copies and explain when
each one is the right choice.

---

## Exercises

Work through these in order; each has a concrete observation to make.

1. **Latency, not bandwidth.** Run `BM_Sequential/4096` and `/1048576`.
   Compute the implied per-request latency for each and explain the 50× gap.
2. **Break the contract.** Write a tiny program that reads a file with
   O_DIRECT at offset 1, then at offset 4096 with a buffer allocated at
   `data()+1`. Observe `EINVAL` both times, then fix each with
   `align_up`/`fp::AlignedBuffer`.
3. **Depth sweep.** Run `BM_Queued` and find where throughput stops improving.
   Predict what the thread-pool fallback will do at the same depths, then
   check.
4. **Shard size.** Build a 1 GiB-shard layout (`DSIO_BENCH_SHARDS=…`) and
   compare with the 32 MiB default. Explain the difference with Lesson 4's
   ramp table.
5. **Cache illusion.** Read a 4 GiB file twice with `dd iflag=direct` and once
   buffered; then read 24 GiB of a different file and repeat. Which numbers
   were the page cache?
6. **Memory budget.** Pick a target: 512 MiB resident, 4 MiB batches. Choose
   `prefetch` and `read_depth`, then verify with
   `BM_DatasetMatrix` that throughput stays within ~10% of the best.
7. **GPU plan.** On this machine, call `gpu_direct_available()` and
   `open_device_sink()`; write the fallback branch you would ship. Then sketch
   (in prose) what changes when the answer is `true`.

## Glossary

| Term | Meaning |
|---|---|
| **page cache** | kernel RAM holding recently read file data; fast, but a copy and it evicts other data |
| **O_DIRECT** | open flag: bypass the page cache, DMA into your buffer; requires block-aligned buffers/offsets/lengths |
| **block size** | the granularity the device demands for direct I/O; logical (512) vs filesystem (`statfs`, 4096 here) |
| **io_uring** | Linux async I/O interface: a submission queue and a completion queue, shared with the kernel |
| **queue depth** | how many requests are in flight at once; converts latency into bandwidth |
| **shard** | a byte range of a file; the unit of dataset layout, parallel reads and shuffling |
| **manifest** | the TSV file listing shards (`path`, `offset`, `length`) |
| **batch** | the bytes handed to one callback; the unit ml turns into records |
| **prefetch** | batches the pipeline reads ahead of the consumer |
| **read depth** | segment reads in flight inside the pipeline (per shard) |
| **epoch** | one full pass over the dataset; shuffling happens per epoch |
| **DMA** | direct memory access: a device writes memory without the CPU copying |
| **GPUDirect Storage (cuFile)** | NVIDIA's path for DMA from NVMe straight into GPU memory; needs `nvidia-fs` |
| **staging** | reading into host memory first, then copying to the device |
| **vhdx** | the virtual disk WSL2 keeps on the Windows host — the layer that caps throughput here |
