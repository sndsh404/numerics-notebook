# Exercises

Each file here is a topic from the main package with the function bodies removed. The docstrings state the task. The reference implementations live in `calccode/`.

The study loop:

1. Read the matching note in `notes/`.
2. Open the exercise file and implement the functions.
3. Run the exercise tests:

   ```bash
   python -m pytest tests/test_exercises.py --run-exercises
   ```

4. Compare your version against the reference in `calccode/`. Yours can differ in style; it should agree in numbers.

The exercise tests skip by default so the main suite stays green while exercises are unsolved.
