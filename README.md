# Restore TensorFlow

Research archive on TensorFlow's history: the 2015 open-source release, the C++/Python split, DistBelief origins, the define-and-run to eager execution shift in TF 2.0, XLA/HLO/Pallas/Mosaic internals, Visual Blocks, and the question of restoring direct chip-level dataflow programming. Includes findings on Google's standardized algorithm visualizations, formula-flow diagrams, and the analog-computing angle for bypassing OS abstraction layers.

## Core Questions

- How did TensorFlow's execution model evolve from static graphs (TF 1.x) to eager execution (TF 2.0)?
- What is the actual interface Google's TPU engineers use — XLA, HLO, Pallas, Mosaic, LLO, tpuasm?
- Can we restore a more direct, dataflow-oriented programming model that feels closer to the hardware?
- Is there an analog or continuous-signal approach to using digital chips that the OS abstraction layers have buried?
- Where do Google's standardized algorithm visualizations and formula-flow diagrams live, and can they map onto Max MSP / Pure Data style patching?

## Key Findings So Far

- TensorFlow shipped November 9, 2015. Python was first-class from day one alongside a C++ API. The core was always C++ and CUDA.
- TF 1.x was define-and-run: static graph built in Python, executed through a Session.
- TF 2.0 (2019) flipped to eager execution by default, adopted Keras as the official high-level API. Driven by PyTorch eating research-paper share and internal Googler friction.
- Pallas: write a Python function taking Refs (memory references staged in TPU on-chip VMEM), explicitly control DMA transfers between HBM and VMEM, software pipelining, tile grids across TensorCores. GPU path lowers through Mosaic GPU (basically Triton). TPU path lowers through Mosaic, emitting LLO for TensorCore. `interpret=True` runs as pure NumPy on CPU. `tpuasm` disassembles machine listings byte-for-byte.
- XLA (Accelerated Linear Algebra): tracer captures Python/JAX into HLO intermediate representation, fuses ops, plans memory layouts, emits systolic-array code. Jeff Dean's framing: "bolting jet engines onto a tank."
- Visual Blocks for ML (formerly Rapsai): node-graph editor, drag and drop, TensorFlow.js under the hood, real-time preview. Prototyping/outreach tool, not the internal workflow.
- C++ API lives under `tensorflow/cc`, C API is the stable base everything binds to.
- AMD Instinct MI430X (announced Nov 12 at SC26): native FP64 at 288 TFLOPS, 432 GB HBM4, 23 TB/s bandwidth. Emulated FP64 needs ~3x memory of native.
- AMD MI455X: 21,300 tokens/sec/GPU on DeepSeek-R1, ~30% ahead of Nvidia Vera Rubin (AMD's own measurement, not peer-reviewed).

## Open Threads

- Microspan deintegration: separation quality sweep from 4ms to 128ms against window length, SI-SDR metrics.
- Carrier-tone time synthesis and telephony as channel theory.
- Analog-style programming of digital chips to bypass OS abstraction layers.
- Mapping Google's formula-flow visualizations onto modular DSP patching paradigms.
- Neural computation shift from spike-as-bit to population codes and dynamical systems.

## Related Repos

- https://github.com/arccoxx/voice-switching-psychophysics
- https://github.com/arccoxx/microspan-deintegration (planned)
- https://github.com/arccoxx/timemachines
- https://github.com/arccoxx/blacksite