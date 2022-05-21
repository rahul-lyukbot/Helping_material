def sum_of_digit(n):
  assert n >= 0 and n == int(n), "number must be a positive integer"
  if n == 0:
    return 0
  else:
    return (int(n%10)) + sum_of_digit(int(n/10))
