# Hints - Challenge 2

Reveal these one at a time - try each function before the next hint.

## Hint 1: to_tensor
`torch.tensor(data, dtype=torch.float32)`. Passing the `dtype` is what makes
integer input come out as floats.

## Hint 2: tensor_info
Read the attributes off the tensor: `tuple(t.shape)`, `t.ndim`, `t.numel()`, and
`str(t.dtype)`. Build the dict with exactly the keys `shape`, `ndim`, `numel`,
`dtype`.

## Hint 3: reshape
`t.reshape(rows, cols)`. (`t.view(rows, cols)` also works for contiguous tensors.)

## Hint 4: get_column
Slice all rows of one column: `t[:, j]`. The `:` keeps every row, so the result
is 1-D with length equal to the number of rows.

## Hint 5: row_sums
`t.sum(dim=1)`. `dim=1` reduces along the columns, leaving one sum per row.
(`dim=0` would sum down each column instead - a common mix-up.)

## Hint 6: add_vector_to_rows
Just `matrix + vector`. Broadcasting stretches the length-`n` vector across all
`m` rows automatically; no loop needed.

## Hint 7: matmul
`a @ b` (or `torch.matmul(a, b)`). Do **not** use `*` - that's element-wise and
will give the wrong answer (and often a shape error).

## Hint 8: Debugging a values test
If a values test fails, print your result next to the expected tensor - a
`dim=0`/`dim=1` swap or a `*`-vs-`@` mix-up is usually the cause.
