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
h = 0.1;

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
plot(times_cont, solution_ODE(times_cont), 'k--', 'Linewidth', 2)
hold on


%------------
% Forward Euler method.
%------------

disp('Processing forward Euler method.')

% Implement the method.
for n = 1:N
    x(n+1) = x(n) + h * f((n-1) * h, x(n));
end

% Plot the outcome.
plot(times, x, 'Linewidth', 2)
hold on

% Compute the average error.
display(strcat(...
    'Mean error = ', ...
    num2str(mean(abs(x - solution_ODE(times))))...
))
    

%------------
% Backward Euler method.
%------------

disp('Processing backward Euler method.')

y = x(1);
for n = 1:N
    
    % Define F(y) such that F(y)=0 is equivalent to solving
    % the implicit equation that arises in each iteration of
    % backward Euler method. Implement a Newton-Raphson method
    % to compute x(n+1). Start the NR iteratiion from the explicit 
    % Euler solution. Iterate till | F(y) | > 10^{-5}
    
    % Insert your code here to approximately solve F(y) = 0.
    F = @(ynp1) ynp1-y-h*f(h*(n+1), ynp1);
    F_prime = @(ynp1) 1-h*f_prime(h*(n+1), ynp1);
    y = newton_raphson(y, F, F_prime, 1e-5, 1000);
    x(n+1) = y;
end


% Plot the outcome.
plot(times, x, 'Linewidth', 2)
hold on

% Compute the average error.
display(strcat(...
    'Mean error = ', ...
    num2str(mean(abs(x - solution_ODE(times))))...
))
   

%------------
% Trapezoidal method.
%------------

disp('Processing trapezoidal method.')
y = x(1);
for n = 1:N
    
    % Define F(y) such that F(y)=0 is equivalent to solving
    % the implicit equation that arises in each iteration of
    % the trapezoidal method. Implement a Newton-Raphson method
    % to compute x(n+1). Start the NR iteratiion from the explicit 
    % Euler solution. Iterate till | F(y) | > 10^{-5}
    
    % Insert your code here to approximately solve F(y) = 0.
    F = @(ynp1) ynp1-y-(h/2)*(f(h*n, y)+f(h*(n+1), ynp1));
    F_prime = @(ynp1) 1-0.5*h*f_prime(h*(n+1), ynp1);
    y = newton_raphson(y, F, F_prime, 1e-5, 1000);
    x(n+1) = y;
    
end

% Plot the outcome.
plot(times, x, 'Linewidth', 2)
xlabel("t")
ylabel("x")
title("Comparison of several integration schemes")
hold on

% Compute the average error.
display(strcat(...
    'Mean error = ', ...
    num2str(mean(abs(x - solution_ODE(times))))...
))
   
%------------
% Add legends, grid to the plot.
%------------

legend({'Analytical', 'Forward Euler', 'Backward Euler', 'Trapezoidal'}, 'FontSize',14)
grid on
hold off

