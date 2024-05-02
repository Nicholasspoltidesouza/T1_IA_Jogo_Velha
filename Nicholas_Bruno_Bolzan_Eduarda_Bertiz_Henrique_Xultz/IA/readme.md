Install python

dependencies:

`pip install scikit-learn`
`pip install fastapi uvicorn`
`pip install uvicorn`


run:
`python .\starter.py`

Routes
its get with the board game after /
all routes below have an example with a final board that O wins
tree:
`http://localhost:8000/verifyTree/o,o,o,x,x,b,x,o,x`
MLP:
`http://localhost:8000/verifyMLP/o,o,o,x,x,b,x,o,x`
KNN:
`http://localhost:8000/verifyKNN/o,o,o,x,x,b,x,o,x`
