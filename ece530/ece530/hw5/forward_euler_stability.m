clc
clear
close all

% Code for HW 5, ECE 530, Fall 2023.

% Consider the ODE \dot{x} = f(t, x), starting from x0.
% Define the function f.
L = -5;
f = @(t, x) (L*x + (1-L) * cos(t) - (1 + L) * sin(t));

% Define the derivative of 'f' with respect to 'x'.
deriv_f = @(t,x) (L);
f_prime = deriv_f;

% Initial point
x0 = 1;

% Time horizon
T = 10;

% Step-size
hs = [0.15, 0.3, 0.45];
h=hs(1);

% Number of iterations
N = floor(T/h);
times = (0:h:N*h)';

% Create a vector of results in x. Notice that our implementation is 
% such that x(1) = x_0, x(2) = x_1, x(3) = x_2, etc.
x = zeros(1 + N, 1);
x(1) = x0;


%----------------------
% Analytical Solution of ODE
%----------------------

disp('Processing analytical solution.')

% Define the analytical solution.
solution_ODE = @(t) (cos(t) + sin(t));

% Plot the analytical solution.
times_cont = (0:0.01:T)';
figure(1);
plot(times_cont, solution_ODE(times_cont), 'k--', 'Linewidth', 2)
hold on


%------------
% Forward Euler method.
%------------

disp('Processing forward Euler method.')
for i = 1:3
  h = hs(i);
  N = floor(T/h);
  times = (0:h:N*h)';

  % Create a vector of results in x. Notice that our implementation is 
  % such that x(1) = x_0, x(2) = x_1, x(3) = x_2, etc.
  x = zeros(1 + N, 1);
  x(1) = x0;
  % Implement the method.
  for n = 1:N
      x(n+1) = x(n) + h * f((n-1) * h, x(n));
  end
  % Plot the outcome.
  plot(times, x, 'Linewidth', 2)
  xlabel("t")
  ylabel("x")
  title("Comparison of several step sizes for Forward Euler integration")
  hold on

  % Compute the average error.
  display(strcat(...
      'Mean error = ', ...
      num2str(mean(abs(x - solution_ODE(times))))...
  ))
endfor


legend({'Analytical', 'Forward Euler, h=0.15', 'Forward Euler, h=0.30', 'Forward Euler, h=0.45'}, 'FontSize',14)
grid on
hold off
print('forward_euler.eps', '-deps', '-color')