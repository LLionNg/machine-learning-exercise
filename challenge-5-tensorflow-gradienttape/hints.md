# Hints - Challenge 5

Reveal these one at a time - try each step before the next hint.

## Hint 1: Seed, then wrap normals in Variables
Start `init_params` with `tf.random.set_seed(seed)`. Then each parameter is
`tf.Variable(tf.random.normal(<shape>))`. The `tf.Variable` wrapper is what makes
it trainable and mutable.

## Hint 2: Match the shapes exactly
`w1 (1, 32)`, `b1 (32,)`, `w2 (32, 16)`, `b2 (16,)`, `w3 (16, 1)`, `b3 (1,)`.
Return them as a list in that order: `[w1, b1, w2, b2, w3, b3]`.

## Hint 3: The forward pass is three lines
```python
x = tf.sigmoid(x @ w1 + b1)
x = tf.sigmoid(x @ w2 + b2)
return x @ w3 + b3
```
`@` is matrix multiply; the bias broadcasts. The last line has **no** sigmoid -
the output must be able to be negative.

## Hint 4: `mse` returns a scalar
`return tf.reduce_mean((pred - target) ** 2)`.

## Hint 5: Record inside the tape, differentiate after
```python
with tf.GradientTape() as tape:
    loss = mse(forward(params, x), y)
grads = tape.gradient(loss, params)
```
Only the `loss` computation goes inside the `with` block. `tape.gradient` returns
one gradient per variable, in the same order you passed `params`.

## Hint 6: Update with `assign_sub`
A `tf.Variable` is mutable - update it in place, downhill:
```python
for v, g in zip(params, grads):
    v.assign_sub(g * lr)      # v <- v - g*lr
```
No `no_grad()` needed: the update is outside the tape, so it isn't recorded.

## Hint 7: No `zero_grad()` here
Unlike PyTorch, a `GradientTape` is fresh every block - gradients don't
accumulate across epochs, so there's nothing to reset. Just make a new tape each
iteration (which the loop already does).

## Hint 8: Return Python floats
The tests expect `list[float]`. Append `float(loss.numpy())`, not the tensor.

## Hint 9: Gradient test failing (a `None` gradient)?
`tape.gradient` returns `None` for any variable the loss doesn't actually depend
on. Make sure all six parameters are used in `forward`, and that the `loss` line
is **inside** the `with tf.GradientTape()` block (ops outside the tape aren't
recorded).

## Hint 10: Loss not going down (or NaN)?
Check `assign_sub` (subtract, i.e. downhill - not `assign_add`), that the output
layer is linear, and that you build a fresh tape each epoch. A wrong update sign
or a too-large learning rate can send the loss to `NaN`.
