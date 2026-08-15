<<<<<<< ours (vime current)
ARG BASE_IMAGE=vllm/vllm-openai:v0.25.1-ubuntu2404
FROM ${BASE_IMAGE}
||||||| base (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
ARG VLLM_IMAGE_TAG=v0.5.13-cu129
FROM slimerl/vllm:${VLLM_IMAGE_TAG} AS vllm
=======
ARG VLLM_IMAGE_TAG=v0.5.15.post1-cu129
FROM slimerl/vllm:${VLLM_IMAGE_TAG} AS vllm
>>>>>>> theirs (slime@2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d translated)

# ======================================== Arguments =============================================

ARG PATCH_VERSION=latest
ARG MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087
ARG DEEPGEMM_COMMIT=b38a77cd193cf38f670caae192310521d24343be
ARG DEEPEP_COMMIT=6845ffd9d59126ec0030c13e0e155935a61e5b5a
ARG DEEPEP_CUDA_ARCH_LIST=
ARG FLASH_QLA_COMMIT=821fd9d37ede18fdc2a4e707fefe3770bfc32e58
ARG TRANSFORMER_ENGINE_COMMIT=c9877beb87ad7e711e1869dd0b5062167ede447a
ARG TRANSFORMER_ENGINE_CUDA_ARCHS=100a;103a
ARG TMS_COMMIT=8d30c59ca12a68d9deccbc9c6599076a1218cbc5

<<<<<<< ours (vime current)
ARG ENABLE_CUDA_13=1
ARG FA2_MAX_JOBS=64
||||||| base (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
ARG ENABLE_CUDA_13=0
ARG TMS_CUDA_MAJOR=
=======
ARG TMS_CUDA_MAJOR=
>>>>>>> theirs (slime@2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d translated)

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

<<<<<<< ours (vime current)
# The validated TransformerEngine 2.16 context-parallel stack uses FA2 + FA3.
RUN pip uninstall -y flash-attn-4 flash_attn_4 || true
RUN MAX_JOBS=${FA2_MAX_JOBS} pip -v install flash-attn==2.8.3 --no-build-isolation

# This FA3 commit provides the window_size_left/window_size_right API used by TE 2.16.
RUN git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention/ && git checkout 002cce0a1068f8c07dfccb5a1d232b9a3276947c && git submodule update --init && \
    cd hopper/ && \
    FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=96 pip -v install . --no-build-isolation && \
    cd /root/ && rm -rf flash-attention/

RUN pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
||||||| base (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
# The VLLM base image may include FA4.  The validated TE 2.16 CP stack uses
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
=======
# CUDA 12 training uses FA2/FA3.  On CUDA 13, keep the FA4 package supplied by
# VLLM instead of replacing its Blackwell inference kernels with FA2.
RUN if [ "$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" = "12" ]; then \
      pip uninstall -y flash-attn-4 flash_attn_4 || true; \
      MAX_JOBS=64 pip -v install flash-attn==2.8.3 --no-build-isolation; \
      git clone https://github.com/Dao-AILab/flash-attention.git && \
      cd flash-attention/ && git checkout 002cce0a1068f8c07dfccb5a1d232b9a3276947c && git submodule update --init && \
      cd hopper/ && \
      FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=96 pip -v install . --no-build-isolation && \
      cd /root/ && rm -rf flash-attention/; \
    fi
>>>>>>> theirs (slime@2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d translated)

RUN pip install flash-linear-attention==0.4.2
<<<<<<< ours (vime current)
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
      pip install nvidia-mathdx==25.6.0 pybind11 ninja wheel packaging && \
      pip -v install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.16; \
||||||| base (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
# FlashQLA: optional GDN backend for Qwen3.5/Qwen3-Next (--qwen-gdn-backend flashqla; requires SM90+)
RUN pip install git+https://github.com/QwenLM/FlashQLA.git --no-build-isolation
RUN pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

# TE does not have wheel on cuda 13 yet, thus need to install from source
RUN if [ "${ENABLE_CUDA_13}" = "1" ]; then \
      pip install nvidia-mathdx==26.6.0 && \
      pip -v install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.16; \
=======
# FlashQLA currently requires TileLang 0.1.9, while VLLM 0.5.15.post1 on
# CUDA 13 requires 0.1.11.  Keep the existing CUDA 12 installation and use
# the default FLA backend on CUDA 13 instead of downgrading VLLM's runtime.
RUN if [ "$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" = "12" ]; then \
      pip install \
        git+https://github.com/QwenLM/FlashQLA.git@${FLASH_QLA_COMMIT} \
        --no-build-isolation; \
      pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/; \
    fi

# The cu13 TE wheel is built against a newer cuBLAS than VLLM's CUDA 13.0
# toolkit.  Build TE against the base image's libraries on CUDA 13; retain the
# validated wheel path for CUDA 12.
RUN if [ "$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" = "13" ]; then \
      NVTE_CUDA_ARCHS="${TRANSFORMER_ENGINE_CUDA_ARCHS}" MAX_JOBS=64 \
        pip -v install --no-build-isolation \
        git+https://github.com/NVIDIA/TransformerEngine.git@${TRANSFORMER_ENGINE_COMMIT}; \
>>>>>>> theirs (slime@2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d translated)
    else \
      pip -v install --no-build-isolation "transformer_engine[pytorch]==2.16.1"; \
    fi

RUN NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

RUN git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
    cd Megatron-LM && git checkout ${MEGATRON_COMMIT}

<<<<<<< ours (vime current)
# torch_memory_saver pinned to a193d9dd (upstream slime #1916).
# TMS_CUDA_MAJOR is required by this pin's build backend for CUDA wheels;
# auto-detect from the running torch's CUDA major (12 for cu129, 13 for cu130).
RUN TMS_CUDA_MAJOR="$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')" && \
||||||| base (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
RUN TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}" && \
=======
# zhuzilin fork builds, grouped together right after Megatron-LM:
# torch_memory_saver, plus the GLM-5 train/rollout alignment kernels
# (DeepGEMM batch-invariant selection and the DeepEP low-latency fork).
RUN TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}" && \
>>>>>>> theirs (slime@2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d translated)
    export TMS_CUDA_MAJOR && \
<<<<<<< ours (vime current)
    pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 --no-cache-dir --force-reinstall
ARG MEGATRON_BRIDGE_COMMIT=07d61e1547a8356cc34928f7eb20226d2f9db3fa
RUN git clone https://github.com/radixark/Megatron-Bridge.git && \
    cd Megatron-Bridge && git checkout ${MEGATRON_BRIDGE_COMMIT}
COPY docker/patch/${PATCH_VERSION}/megatron_bridge.patch /root/Megatron-Bridge/
RUN cd Megatron-Bridge && \
    git apply megatron_bridge.patch --3way && \
    if grep -R -n '^<<<<<<< ' .; then \
      echo "Patch failed to apply cleanly. Please resolve conflicts." && \
      exit 1; \
    fi && \
    rm megatron_bridge.patch && \
    pip install . --no-deps --no-build-isolation
RUN pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation
||||||| base (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
    pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 --no-cache-dir --force-reinstall
RUN pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
RUN pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation
=======
    pip install git+https://github.com/zhuzilin/torch_memory_saver.git@${TMS_COMMIT} --no-cache-dir --force-reinstall

# DeepGEMM's batch-invariant kernel selection; build the VLLM-compatible wheel.
RUN git clone https://github.com/zhuzilin/DeepGEMM.git --recursive && \
    cd DeepGEMM && git checkout ${DEEPGEMM_COMMIT} && \
    git submodule update --init --recursive && \
    bash build_sgl_deep_gemm.sh && \
    pip install --force-reinstall --no-deps dist/sgl_deep_gemm-*.whl && \
    cd /root/ && rm -rf DeepGEMM

# DeepEP low-latency alignment fork.  CUDA 13 relocated the CCCL/libcu++ headers
# (<cuda/std/*>) under include/cccl, which nvshmem_tensor.h includes, so add that
# directory to the compile search path (nvcc + host compiler) when it exists.
RUN git clone https://github.com/zhuzilin/DeepEP.git /root/DeepEP && \
    cd /root/DeepEP && git checkout ${DEEPEP_COMMIT} && \
    CUDA_ARCHS="${DEEPEP_CUDA_ARCH_LIST:-9.0}" && \
    if [ -d /usr/local/cuda/include/cccl ]; then \
      export CPATH="/usr/local/cuda/include/cccl${CPATH:+:$CPATH}"; \
      export NVCC_PREPEND_FLAGS="-I/usr/local/cuda/include/cccl ${NVCC_PREPEND_FLAGS:-}"; \
    fi && \
    TORCH_CUDA_ARCH_LIST="${CUDA_ARCHS}" MAX_JOBS=64 python setup.py bdist_wheel && \
    pip install --force-reinstall --no-deps dist/deep_ep-*.whl && \
    cd /root/ && rm -rf DeepEP
>>>>>>> theirs (slime@2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d translated)

<<<<<<< ours (vime current)
||||||| base (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
# This patch from masahi will be included in later Triton releases
RUN if [ "$ENABLE_CUDA_13" = "1" ]; then \
    (cd /root && git clone -b feat/v350_plus_8045 https://github.com/fzyzcjy/triton.git && cd triton && pip install -r python/requirements.txt && pip install --verbose -e .); \
  fi

=======
RUN pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

>>>>>>> theirs (slime@2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d translated)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --ignore-installed PyJWT && \
    pip install -r /tmp/requirements.txt

<<<<<<< ours (vime current)
# https://github.com/pytorch/pytorch/issues/168167
# cu130 base already ships nvidia-cudnn-cu13 9.19.0; only cu129 needs the pin.
RUN if [ "${ENABLE_CUDA_13}" != "1" ]; then \
      pip install nvidia-cudnn-cu12==9.16.0.29; \
    fi

# reinstall numpy 1.x for megatron
RUN pip install "numpy<2" "scipy<1.18"
||||||| base (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
# Temporarily install another sgl-kernel version for GB300 without rebuilding the whole image
RUN if [ "$ENABLE_CUDA_13" = "1" ]; then \
    SGL_KERNEL_VERSION=0.3.17.post2 && \
    python3 -m pip install https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sgl_kernel-${SGL_KERNEL_VERSION}+cu130-cp310-abi3-manylinux2014_$(uname -m).whl --force-reinstall --no-deps; \
  fi

# reinstall numpy 1.x for megatron
RUN pip install "numpy<2"

RUN pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/vllm_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
RUN python -c "import vllm_router; assert 'slime' in vllm_router.__version__"
=======
# Megatron currently requires NumPy 1.x, while SciPy 1.18+ requires NumPy 2.x.
# Pin both packages together so downgrading NumPy does not leave an incompatible SciPy.
RUN pip install "numpy==1.26.4" "scipy==1.17.1"

RUN pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-9daabcd/vllm_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
RUN python -c "import vllm_router; assert 'slime' in vllm_router.__version__"
>>>>>>> theirs (slime@2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d translated)

RUN rm -rf /root/.cache/pip /root/flash-attention

# ====================================== Patches ============================================

COPY docker/patch/${PATCH_VERSION}/megatron*.patch /root/Megatron-LM/
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
||||||| base (slime@680824dd5e01a2e83750bf87fc366ec6fa98766c translated)
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
=======
    if [ -f megatron-vllm-aligned.patch ]; then \
      git apply megatron-vllm-aligned.patch --3way && \
      if grep -R -n '^<<<<<<< ' .; then \
        echo "megatron-vllm-aligned.patch failed to apply cleanly. Please resolve conflicts." && \
        exit 1; \
      fi; \
    fi && \
    rm -f megatron*.patch

# TODO temporarily skip patching for GB200/GB300 (and require users to bring their own vllm version). should add back later.
ARG ENABLE_VLLM_PATCH=1
COPY docker/patch/${PATCH_VERSION}/vllm*.patch /sgl-workspace/vllm/
RUN if [ "$ENABLE_VLLM_PATCH" = "1" ]; then \
  cd /sgl-workspace/vllm && \
  for patch in vllm.patch vllm-top_p.patch vllm-release_hicache.patch vllm-pull_weights.patch vllm-deterministic.patch; do \
    if [ ! -f "$patch" ]; then \
      continue; \
    fi; \
    if ! git apply --check "$patch"; then \
      echo "Patch $patch does not apply to the VLLM source in the base image."; \
      exit 1; \
    fi; \
    git apply "$patch" || exit 1; \
  done && \
  rm vllm*.patch; \
fi
>>>>>>> theirs (slime@2fa9a442f2f4d4e6ec4041fe110e0319af56ba4d translated)

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
# and `ray job submit` append to `vllm serve` and break.
ENTRYPOINT []
CMD ["/bin/bash"]
