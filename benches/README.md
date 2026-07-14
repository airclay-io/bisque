# Processor Benchmarks

Run the complete Criterion suite with:

```sh
cargo bench --bench processors --all-features
```

The benchmarks use stereo `f32` audio at 48 kHz and cover block sizes 1, 32,
64, and 512. Criterion reports time per block and throughput in rendered
frame-channel elements per second.

Input creation and processor preparation are excluded from timing. In-place
processors receive a fresh input block for every measured call, while their
internal state and `sample_pos` continue as they would in a stream. Event
arrays and sidechain input are also prepared outside the timed routine.

Results depend on the machine, toolchain, power settings, and background load.
Use Criterion's local baselines to compare changes on the same system:

```sh
cargo bench --bench processors --all-features -- --save-baseline before
cargo bench --bench processors --all-features -- --baseline before
```

These results provide performance evidence rather than CI pass/fail thresholds.
Allocation behavior is checked separately by the no-allocation integration
tests.
