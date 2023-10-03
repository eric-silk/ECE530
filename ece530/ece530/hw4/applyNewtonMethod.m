% Code to run a generic Newton method.
clear all
close all
clc; history -c

MAX_ITER = 100;

% Function and its gradient and hessian.

f = @(x, y) (cos(x.^2 - 3*y)  + sin(x.^2 + y.^2));

gradf = @(x, y) ([ 2*x*cos(x^2 + y^2) - 2*x*sin(x^2 - 3*y); ...
                    3*sin(x^2 - 3*y) + 2*y*cos(x^2 + y^2) ...
                ]);
            
hessf = @(x,y) [ 2*cos(x^2 + y^2) - 2*sin(x^2 - 3*y) - 4*x^2*cos(x^2 - 3*y) - 4*x^2*sin(x^2 + y^2), ...
                    6*x*cos(x^2 - 3*y) - 4*x*y*sin(x^2 + y^2); ...
                 6*x*cos(x^2 - 3*y) - 4*x*y*sin(x^2 + y^2), ...
                    2*cos(x^2 + y^2) - 9*cos(x^2 - 3*y) - 4*y^2*sin(x^2 + y^2)];
 
% Plot the function and its contour plot to
% appreciate how this function looks like
[Xgrid, Ygrid] = meshgrid(-1.5:0.1:1.5, -1.5:0.1:1.5);
Z = f(Xgrid, Ygrid);

subplot(2,1,1), contour(Xgrid, Ygrid, Z, 40,'Linewidth',2),
grid on, xlabel('$x_1$', 'Interpreter','Latex', 'Fontsize', 20), 
ylabel('$x_2$', 'Interpreter','Latex', 'Fontsize', 20),

subplot(2,1,2), surf(Xgrid, Ygrid, Z), grid on,
xlabel('$x_1$', 'Interpreter','Latex', 'Fontsize', 20), 
ylabel('$x_2$', 'Interpreter','Latex', 'Fontsize', 20),
zlabel('$f(x_1, x_2)$', 'Interpreter','Latex', 'Fontsize', 20),


                
% Initialize at (1.2, 0.5).
xk = [1.2; 0.5];

% Tolerance for the gradient value.
tolerance = 1e-6;

% Variables required for the iteration.
shouldIterate = true;
iterationK = 1;
errorGF = 0
while (shouldIterate)
    
    % Display the current iteration.
    display(strcat('Iteration # ', num2str(iterationK)))
    display(strcat( ...
        'Current values of (x,y) = [', ...
        num2str(xk'), ...
        ']'...
    ))
    % Compute the norm of the gradient.
    errorGF = norm(gradf(xk(1), xk(2)), 2);
    
    display(strcat( ...
        'Current norm of gradient = ', ...
        num2str(errorGF) ...
    ))
    % Added by ESilk to display if the Hessian is PD!
    if(iterationK == 1)
        Hk = hessf(xk(1), xk(2));
        output_str = {"False", "True"};
        H_is_PD = "False";
        if(all(eig(Hk)>0))
          H_is_PD = "True";
        end
        printf('H > 0: %s', H_is_PD)
        disp('')
    end
    %pause
        % Uncomment the previous line if you want to
        % look at each new iterate.
        
    if (errorGF > tolerance)
        
        % Compute the current Hessian
        Hk = hessf(xk(1), xk(2));
        Hk = modifyHessian(Hk);
        if(iterationK == 1)
            display('Modified H0:')
            display(num2str(Hk))
        end
        display('')
        assert (all(eig(Hk) > 0))
            % Problem 4.1: Uncomment the two lines above and include your function
            % to modify the Hessian as per your answer to part (b).
            
            
        % Compute the Newton direction.
        sk = - inv(Hk) * gradf(xk(1), xk(2));
        
        alphak = 1;
        
        % alphak = lineSearch(f, gradf, xk, sk, 0.0001);
        % if alphak == -1
        %    break
        % end
            % Probelm 4.2: Uncomment the above lines for implementing your own 
            % line search function. It also enforces that the line
            % search in fact converges.
            
        % Compute the next iterate.
        xk = xk + alphak * sk;
        
        iterationK = iterationK + 1;
    else
        % Error is within tolerance
        shouldIterate = false;
    end

    if iterationK == MAX_ITER
        display('Did not converge within 100 iterations')
        break
    end
end

display(strcat( ...
    'Function value at last iterate = ', ...
    num2str(f(xk(1), xk(2))) ...
))
Hk = hessf(xk(1), xk(2));

H_is_PD = "False";
if(all(eig(Hk)>=0))
  H_is_PD = "True";
end

display(strcat('Iterate is a local minimum:', ...
  H_is_PD
 ))