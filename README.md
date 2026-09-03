This repo houses the code used to generate the lynx dataset.

The following guide will explain how to reproduce the dataset creation process.

# 1. Download the genome

```bash
# hg38 reference FASTA (UCSC)
curl -L \
  https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz \
  -o data/hg38.fa.gz
gunzip -k data/hg38.fa.gz
samtools faidx hg38.fa
```

# 2. Generate the non-overlapping 262,144-bp windows

This script finds 262,144-bp windows that are:
- non-overlapping
- contain no unknown basepairs
- contain less than 50% of basepairs that are either unmappable (data/unmap_macro.bed) or on the ENCODE blacklist (data/hg38-blacklist.v2.bed.gz)

```bash
uv run generate_hg38_regions.py \
  --fasta data/hg38.fa \
  --unmappable data/unmap_macro.bed \
  --blacklist data/hg38-blacklist.v2.bed.gz
```

# 3. Download the raw experimental tracks

This will take some time...

```bash
uv run download_raw_tracks.py
```

# 4. Generate shards.jsonl

```bash
uv run build_shards_jsonl.py \
  --regions genomic_regions.bed \
  --output shards.jsonl \
  --target-shards 1000 \
  --seed 71798669716578
```

# 5. Process shards on slurm cluster

If you don't have slurm cluster, you should be able to easily reverse engineer this to run without slurm.

```bash
bash launch_shards.sh \
  --account stf \
  --partition compute
```

Please check if there's any remaining partial files in the LYNX_DATASET. If there is, then run the launch shards script again and it will attempt to resume what's missing.