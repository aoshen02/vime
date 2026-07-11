ARG BASE_IMAGE=vllm/vllm-openai:v0.23.0-cu129-ubuntu2404
FROM ${BASE_IMAGE}

# ======================================== Arguments =============================================

ARG PATCH_VERSION=latest
ARG MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087

ARG ENABLE_CUDA_13=0

# ======================================== Setup =============================================

WORKDIR /root/

# ======================================== Apt dependencies =============================================

# vllm/vllm-openai base is an inference image — add CUDA dev headers + cmake/git
# so TE / apex / flash-attn source builds find cusparse.h etc.
# Dev packages must match the base toolkit: -12-9 for cu129, -13-0 for cu130.
RUN apt-get update && apt-get install -y \
    nvtop rsync dnsutils prometheus git cmake && \
    if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      apt-get install -y \
        cuda-nvrtc-dev-13-0 cuda-nvml-dev-13-0 cuda-profiler-api-13-0 cuda-nvtx-13-0 \
        libcusparse-dev-13-0 libcusolver-dev-13-0 libcufft-dev-13-0 libcurand-dev-13-0 \
        libcudnn9-dev-cuda-13; \
    else \
      apt-get install -y \
        cuda-nvrtc-dev-12-9 cuda-nvml-dev-12-9 cuda-profiler-api-12-9 cuda-nvtx-12-9 \
        libcusparse-dev-12-9 libcusolver-dev-12-9 libcufft-dev-12-9 libcurand-dev-12-9 \
        libcudnn9-dev-cuda-12; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# vllm/vllm-openai base only ships python3; subsequent source builds invoke `python`.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# ====================================== Python dependencies ============================================

# The vLLM base image may include FA4.  The validated TE 2.16 CP stack uses
# FA2 + FA3, so remove FA4 explicitly to avoid backend selection conflicts.
RUN pip uninstall -y flash-attn-4 flash_attn_4 || true

# The compilation is slow, thus should be put at top
RUN MAX_JOBS=64 pip -v install flash-attn==2.8.3 --no-build-isolation

# The compilation is slow, thus should be put at top.  This FA3 commit uses the
# window_size_left/window_size_right API required by TransformerEngine 2.16 CP.
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/ && git checkout 002cce0a1068f8c07dfccb5a1d232b9a3276947c && git submodule update --init && \
    cd hopper/ && \
    FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=96 pip -v install . --no-build-isolation && \
    cd /root/ && rm -rf flash-attention/

RUN pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps

RUN pip install flash-linear-attention==0.4.2
# FlashQLA: optional GDN backend for Qwen3.5/Qwen3-Next (--qwen-gdn-backend flashqla; requires SM90+)
ARG INSTALL_FLASHQLA=0
RUN if [ "${INSTALL_FLASHQLA}" = "1" ]; then \
      pip install git+https://github.com/QwenLM/FlashQLA.git --no-build-isolation; \
    else \
      echo "Skipping FlashQLA (INSTALL_FLASHQLA=0; sm90/Hopper-only — use --qwen-gdn-backend fla)"; \
    fi
RUN pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

# cublas dev header for TE CMake (arm64 base ships runtime .so but not the header).
# cu13 also needs the -13-0 headers that TE's nvcc build expects.
RUN apt-get update && \
    if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      apt-get install -y libcublas-dev-13-0; \
    else \
      apt-get install -y libcublas-dev-12-9; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# TE does not publish a cu13 wheel; build from source when ENABLE_CUDA_13=1.
RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then \
<<<<<<< ours (vime current)
      pip install nvidia-mathdx pybind11 ninja wheel packaging && \
      pip -v install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.10; \
||||||| base (slime@a897e1f4 translated)
      pip install nvidia-mathdx==26.6.0 && \
      pip -v install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.10; \
=======
      pip install nvidia-mathdx==26.6.0 && \
      pip -v install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.16; \
>>>>>>> theirs (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
    else \
      pip -v install --no-build-isolation "transformer_engine[pytorch]==2.16.1"; \
    fi

RUN NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && git checkout ${MEGATRON_COMMIT}

# torch_memory_saver pinned to a193d9dd (upstream slime #1916).
# TMS_CUDA_MAJOR is required by this pin's build backend for CUDA wheels;
# auto-detect from the running torch's CUDA major (12 for cu129, 13 for cu130).
RUN TMS_CUDA_MAJOR="$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" && \
    export TMS_CUDA_MAJOR && \
    pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 --no-cache-dir --force-reinstall
RUN pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
RUN pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

COPY requirements.txt /tmp/requirements.txt
RUN pip install --ignore-installed PyJWT && \
    pip install -r /tmp/requirements.txt

<<<<<<< ours (vime current)
# https://github.com/pytorch/pytorch/issues/168167
# cu130 base already ships nvidia-cudnn-cu13 9.19.0; only cu129 needs the pin.
RUN if [ "${ENABLE_CUDA_13}" != "1" ]; then \
      pip install nvidia-cudnn-cu12==9.16.0.29; \
    fi
||||||| base (slime@a897e1f4 translated)
# Temporarily install another sgl-kernel version for GB300 without rebuilding the whole image
RUN if [ "$ENABLE_CUDA_13" = "1" ]; then \
    SGL_KERNEL_VERSION=0.3.17.post2 && \
    python3 -m pip install https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu130-cp310-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps; \
  fi

# https://github.com/pytorch/pytorch/issues/168167
RUN pip install nvidia-cudnn-cu12==9.16.0.29

# reinstall numpy 1.x for megatron
RUN pip install "numpy<2"
=======
# Temporarily install another sgl-kernel version for GB300 without rebuilding the whole image
RUN if [ "$ENABLE_CUDA_13" = "1" ]; then \
    SGL_KERNEL_VERSION=0.3.17.post2 && \
    python3 -m pip install https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu130-cp310-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps; \
  fi

# reinstall numpy 1.x for megatron
RUN pip install "numpy<2"
>>>>>>> theirs (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)

<<<<<<< ours (vime current)
# reinstall numpy 1.x for megatron; pin scipy<1.18 alongside it. vime's vllm/vllm-openai
# base ships NO scipy, so unpinned it pulls scipy>=1.18, which hard-requires numpy>=2 and
# uses np.long (removed in numpy>=1.24) -> AttributeError against the numpy<2 above ->
# `import scipy/transformers` crash. slime's sglang base resolves scipy to 1.17.1 (numpy-1.x
# compatible) natively, so slime needs no scipy pin -- this is a vime base-image divergence.
# Real boundary is 1.18 (slime runs 1.17.1 fine), not the earlier 1.14 guess.
RUN pip install "numpy<2" "scipy<1.18"
||||||| base (slime@a897e1f4 translated)
RUN pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-1117d05/vllm_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
RUN python -c "import vllm_router; assert 'vime' in vllm_router.__version__"
=======
RUN pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/vllm_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
RUN python -c "import vllm_router; assert 'vime' in vllm_router.__version__"
>>>>>>> theirs (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)

RUN rm -rf /root/.cache/pip /root/flash-attention

# ====================================== Patches ============================================

COPY docker/patch/${PATCH_VERSION}/megatron.patch /root/Megatron-LM/
RUN cd Megatron-LM && \
    git update-index --refresh && \
    git apply megatron.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
      echo "Patch failed to apply cleanly. Please resolve conflicts." && \
      exit 1; \
    fi && \
<<<<<<< ours (vime current)
    rm megatron.patch && \
    pip install -e .

# Patch vLLM with vime's local fixes (see docker/patch/${PATCH_VERSION}/vllm.patch
# for the specifics). vLLM is a pip install (not a git checkout) so apply with
# plain `git apply` (no --3way). --allow-empty keeps the build working once every
# fix has landed upstream and the patch is emptied.
COPY docker/patch/${PATCH_VERSION}/vllm.patch /tmp/vllm.patch
RUN VLLM_SITE="$(python3 -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')" && \
    cd "$VLLM_SITE" && \
    git apply -v --allow-empty /tmp/vllm.patch && \
    rm /tmp/vllm.patch
||||||| base (slime@a897e1f4 translated)
    rm megatron.patch

# TODO temporarily skip patching for GB200/GB300 (and require users to bring their own vllm version). should add back later.
ARG ENABLE_VLLM_PATCH=1
COPY docker/patch/${PATCH_VERSION}/vllm.patch /sgl-workspace/vllm/
RUN if [ "$ENABLE_VLLM_PATCH" = "1" ]; then \
  cd /sgl-workspace/vllm && \
  git update-index --refresh && \
  git apply vllm.patch --3way && \
  if grep -R -n '^<<<<<<< ' .; then \
    echo "Patch failed to apply cleanly. Please resolve conflicts." && \
    exit 1; \
  fi && \
  rm vllm.patch; \
fi

COPY docker/patch/${PATCH_VERSION}/vllm-top_p.patch /sgl-workspace/vllm/
RUN if [ "$ENABLE_VLLM_PATCH" = "1" ]; then \
  cd /sgl-workspace/vllm && \
  git update-index --refresh && \
  git apply vllm-top_p.patch --3way && \
  if grep -R -n '^<<<<<<< ' .; then \
    echo "Patch failed to apply cleanly. Please resolve conflicts." && \
    exit 1; \
  fi && \
  rm vllm-top_p.patch; \
fi
=======
    rm megatron.patch

# TODO temporarily skip patching for GB200/GB300 (and require users to bring their own vllm version). should add back later.
ARG ENABLE_VLLM_PATCH=1
COPY docker/patch/${PATCH_VERSION}/vllm.patch \
     docker/patch/${PATCH_VERSION}/vllm-top_p.patch \
     docker/patch/${PATCH_VERSION}/vllm-release_hicache.patch \
     docker/patch/${PATCH_VERSION}/vllm-pull_weights.patch \
     /sgl-workspace/vllm/
RUN if [ "$ENABLE_VLLM_PATCH" = "1" ]; then \
  cd /sgl-workspace/vllm && \
  git update-index --refresh && \
  for patch in vllm.patch vllm-top_p.patch vllm-release_hicache.patch vllm-pull_weights.patch; do \
    git apply --3way "$patch" || exit 1; \
    if git grep -n '^<<<<<<< ' -- .; then \
      echo "Patch failed to apply cleanly. Please resolve conflicts." && \
      exit 1; \
    fi; \
  done && \
  rm vllm.patch vllm-top_p.patch vllm-release_hicache.patch vllm-pull_weights.patch; \
fi
>>>>>>> theirs (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)

# ====================================== Install main package ============================================

ARG VIME_COMMIT=main
RUN git clone https://github.com/vllm-project/vime.git /root/vime && \
    cd /root/vime && \
    git checkout ${VIME_COMMIT} && \
    pip install -e . --no-deps

RUN cd /root/vime/vime/backends/megatron_utils/kernels/int4_qat && \
    pip install . --no-build-isolation

# Reset ENTRYPOINT inherited from the vllm/vllm-openai base (`vllm serve`), so the
# image is a plain bash/ray environment. Without this, `docker run ... bash -c ...`
# and `ray job submit` append to `vllm serve` and break. Base-image-coupled: slime's
# base has no such entrypoint, so slime's Dockerfile doesn't need this.
ENTRYPOINT []
CMD ["/bin/bash"]
