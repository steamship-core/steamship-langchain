# Load pages of the book
from time import perf_counter

from steamship import Steamship

# Load the package instance stub.

# Invoke the method
t0 = perf_counter()
pkg = Steamship.use(
    "ask-my-book-api",
    version="0.0.2",
)
t1 = perf_counter()

resp = pkg.invoke(
    "generate",
    question="What is specific knowledge"
)
print(t1-t0)
print(perf_counter()-t0)
