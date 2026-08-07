## Sparse Buffers
Sparse buffers reserve a large virtual GPU address range without immediately backing it with physical memory. The KV cache uses 16 of them: one key buffer and one value buffer for each of the model’s 8 full-attention layers.

Heaps provide the physical memory. In this implementation, each heap is 64 MiB and is divided into 256 KiB tiles. Each sparse-buffer page stores K or V projections for 128 tokens, so mapping one page across all 16 buffers consumes:

```text
16 × 256 KiB = 4 MiB
```

A complete 64 MiB heap therefore provides space for:

```text
64 MiB / 4 MiB = 16 page groups
16 × 128 tokens = 2048 tokens
```

When the model needs more KV-cache capacity, `ensure()` maps additional sparse-buffer pages to heap tiles. If the current heap has no remaining tiles, another 64 MiB heap is created and added to the residency set. The sparse buffers do not automatically request memory; the application explicitly grows them before dispatching kernels.
