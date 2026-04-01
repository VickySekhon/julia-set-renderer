# CP431 Term Project
Fractal Image Generation Using `mpi4py` and `OpenGL`.

html<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <img width="214" height="276" src="https://github.com/user-attachments/assets/4179f55b-6e57-43a2-936e-1043cd34aa74" />
  <img width="811" height="437" src="https://github.com/user-attachments/assets/df69e5ce-674f-4694-96ed-2a2a9d44ce7d" />
  <img width="680" height="595" src="https://github.com/user-attachments/assets/1b892fa4-c055-4589-b94e-f47a1a54b7d3" />
  <img width="209" height="237" src="https://github.com/user-attachments/assets/d1b4efed-ee9f-4291-8da9-61ce41cc336b" />
</div>

## Virtual Environment Setup and Dependency Installation

### 1. Create a Virtual Environment
```bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment
- **On Unix**
  ```bash
  source venv/bin/activate
  ```
- **On Windows:**
  ```bash
  venv\Scripts\activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

You can verify the installation by checking that `mpi4py` is installed:
```bash
python3 -c "from mpi4py import MPI; print('MPI4PY installed successfully')"
```

## Commands
### Running Parallel Fractal Generation
- **On Unix**
```bash
mpirun -np <number-of-processors> python3 ./main.py -- <a+bj> <dimension-of-image>
```
- **On Windows**
```bash
mpiexec -n <number-of-processors> python3 ./main.py -- <a+bj> <dimension-of-image>
```
### Running Fractal Renderer
- **On Windows**
```bash
python3 main.py -- <a+bj> <dimension-to-rerender-fractal> <path-to-fractal-npy-array>
```
