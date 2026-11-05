# Third-Party Notices

FlowPilot includes or uses the following third-party projects. Their notices and license terms continue to apply to the relevant components.

## Wan2.2

[`flowpilot/models/vae2_2.py`](flowpilot/models/vae2_2.py) is adapted from the [Wan2.2 VAE implementation](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/vae2_2.py). Copyright 2024–2025 The Alibaba Wan Team Authors. Wan2.2 is licensed under [Apache License 2.0](https://github.com/Wan-Video/Wan2.2/blob/main/LICENSE.txt); a copy is included in this repository as [LICENSE](LICENSE). The VAE weights are downloaded separately; check the terms of the specific weights distribution.

## YOPO and Flightmare

The Docker build fetches `flightgym` from the [ZombieRUNs/YOPO fork](https://github.com/ZombieRUNs/YOPO) at commit `13de2e417d625a3eebcfbe0a8bf97d4316cb7a91`. [YOPO](https://github.com/TJU-Aerial-Robotics/YOPO) is based on [Flightmare](https://github.com/uzh-rpg/flightmare). The upstream projects publish the MIT notices below; verify the notices and any additional terms in the exact fork revision and dependencies when distributing the Docker image. The [standalone renderer](https://zenodo.org/records/5517791) is downloaded separately; its bundled assets and terms should also be checked before redistribution.

### YOPO

MIT License

Copyright (c) 2024, TJU-Aerial-Robotics  
Tianjin University, China

This work is developed based on Flightmare Simulator.  
The original LICENSE can be found in the LICENSE_FLIGHTMARE file.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### Flightmare

MIT License

Copyright (c) 2020 Robotics and Perception Group,  
University of Zurich, Switzerland

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
