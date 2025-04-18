A particle sim with a bunch of profiler-guided optimizations, including:
- A spatial data structure
- GPU acceleration
- Misc GPU data upload/download optimizations
- Aligning data to enable autovectorization
- A thread pool implementation
- An SoA/AoS optimization
- Using a CPU intrinsic
- Frustum culling

On my machine, these optimizations result in:
- 100k particles: 6 ms per frame
- 1M particles: 65 ms per frame

For context, before optimization these numbers were:
- 1k particles: 5 ms per frame
- 10k particles: 190 ms per frame
- 100k particles: frozen

Also supports hot-reloading of shaders and C++ code. (Although I forgot to implement reloading of specifically the fluid sim's compute shaders.)

There are misc minor todos and bugfixes.
The big goals I might get to, when I eventually have time (i.e. never):
- Collision with static objects, and more realistic fluid behavior
- Physically-based surface rendering (currently just draws individual particles)

The code is not particularly well-organized. I posted it so that I could point someone to a specific part of the implementation.

This repo also contains the PDF from a presentation I did about the optimizations.
