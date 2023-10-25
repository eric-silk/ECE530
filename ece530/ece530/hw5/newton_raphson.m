function xn = newton_raphson(x0, func, deriv, tol, max_iter)
  x = x0;
  for n = 1:max_iter
    fx = func(x);
    if (abs(fx) < tol)
      xn = x;
      return;
    endif
    fprime = deriv(x);
    dx = fx/fprime;
    if isnan(dx)
      xn = x;
      return;
    endif
    tmp = x-dx;
    if (isnan(tmp))
  
      xn = x;
      return;
    endif
    x = x - dx;
  endfor
  xn = x0;
  disp("Failed to converge!")
end