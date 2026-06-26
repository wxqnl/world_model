#!/usr/bin/env bash
set -euo pipefail

node_name="${1:?node name required, e.g. node43}"

archive="/0604-10T-test/wm3d_v3_archived_cache_20260608/${node_name}/datasets_cache_wm3d_v3.tar"
node44_archive_zst="/0604-10T-test/wm3d_v3_archived_cache_20260608/node44/datasets_cache_wm3d_v3.tar.zst"
node44_source_host="${V3_SOURCE_HOST:-root@172.27.0.7}"
source_v3="/data/Minko/datasets/cache/wm3d_v3"
shared_v5="/0604-10T-test/wm3d_v5/cache"
base_root="/data/Minko/datasets/cache/wm3d_v5_stage0_base"
base_v3="${base_root}/wm3d_v3"
local_root="/data/Minko/datasets/cache/wm3d_v5_stage0_local"
window_subdir="vggt_window_geom_p64_T16_k8_s4_hw64_full_v1"
base_dirs=(actions rgb_256 vggt_geom qwen_taskemb)
stats_file="action_stats_oxe_droid20k_stage1_world_v1.npz"
rsync_ssh="ssh -i /root/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

copy_dir_from_node44() {
  local dir_name="$1"
  local done_marker="${base_v3}/.${dir_name}.done"
  if [ -f "$done_marker" ]; then
    echo "base_dir_done ${dir_name}"
    return 0
  fi
  mkdir -p "${base_v3}/${dir_name}"
  rsync -a --numeric-ids --partial --info=stats2 -e "$rsync_ssh" \
    "${node44_source_host}:${source_v3}/${dir_name}/" "${base_v3}/${dir_name}/"
  touch "$done_marker"
}

echo "prepare_start $(date) node=${node_name}"
timeout 8 df -h /data /0604-10T-test || true

mkdir -p "$base_root" "$local_root"

need_extract=0
for d in "${base_dirs[@]}"; do
  if [ ! -e "${base_v3}/${d}" ]; then
    need_extract=1
  fi
done
if [ ! -e "${base_v3}/${stats_file}" ]; then
  need_extract=1
fi

if [ "$need_extract" = "1" ]; then
  mkdir -p "$base_v3"
  if [ "$node_name" = "node44" ] && [ -d "$source_v3" ]; then
    echo "link_base_from_local_node44_source $(date)"
    rm -rf "$base_v3"
    ln -s "$source_v3" "$base_v3"
  elif [ -n "$node44_source_host" ]; then
    echo "rsync_base_from_node44_source $(date) host=${node44_source_host}"
    for d in "${base_dirs[@]}"; do
      copy_dir_from_node44 "$d"
    done
    if [ ! -e "${base_v3}/${stats_file}" ]; then
      rsync -a --numeric-ids --partial --info=stats2 -e "$rsync_ssh" \
        "${node44_source_host}:${source_v3}/${stats_file}" "${base_v3}/${stats_file}"
    fi
  elif [ -f "$node44_archive_zst" ]; then
    echo "extract_base_from_node44_zstd_archive $(date)"
    tar -I zstd -C "$base_root" -xf "$node44_archive_zst" \
      wm3d_v3/actions \
      wm3d_v3/rgb_256 \
      wm3d_v3/vggt_geom \
      wm3d_v3/qwen_taskemb \
      "wm3d_v3/${stats_file}"
  elif [ -f "$archive" ]; then
    echo "extract_base_from_node_archive $(date)"
    tar -C "$base_root" -xf "$archive" \
      wm3d_v3/actions \
      wm3d_v3/rgb_256 \
      wm3d_v3/vggt_geom \
      wm3d_v3/qwen_taskemb \
      "wm3d_v3/${stats_file}"
  else
    echo "base_source_missing node44_host=${node44_source_host} node44_archive=${node44_archive_zst} node_archive=${archive}"
    exit 2
  fi
fi

ln -sfn "${base_v3}/actions" "${local_root}/actions"
ln -sfn "${base_v3}/rgb_256" "${local_root}/rgb_256"
ln -sfn "${base_v3}/vggt_geom" "${local_root}/vggt_geom"
ln -sfn "${base_v3}/qwen_taskemb" "${local_root}/qwen_taskemb"
ln -sfn "${base_v3}/${stats_file}" "${local_root}/${stats_file}"

if [ ! -d "${local_root}/${window_subdir}" ]; then
  mkdir -p "${local_root}/${window_subdir}"
fi

echo "copy_window_geom_start $(date)"
window_done="${local_root}/.${window_subdir}.done"
if [ -f "$window_done" ]; then
  echo "window_geom_done ${window_subdir}"
else
  rsync -a --numeric-ids --ignore-existing --info=stats2 --partial \
  "${shared_v5}/${window_subdir}/" "${local_root}/${window_subdir}/"
  touch "$window_done"
fi

echo "prepare_done $(date)"
timeout 8 df -h /data /0604-10T-test || true
