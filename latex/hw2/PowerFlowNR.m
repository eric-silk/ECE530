% Code for HW 2, Problem 2.
% Newton Rhapson iteration for power flow problem

clc;
clear all;
close all;

% Impedance matrix

i = sqrt(-1);
cma = @(x, y) (x * exp(i*y*pi/180));
    % cma yields a complex number with magnitude and angle (in degrees)
    % as arguments to the function.

Y = [   cma(13.1505, -84.7148), cma(3.3260, 93.8141),   cma(9.9504, 95.7106); ...
        cma(3.3260, 93.8141),   cma(13.1505, -84.7148), cma(9.9504, 95.7106); ...
        cma(9.9504, 95.7106),   cma(9.9504, 95.7106),   cma(19.8012, -84.2606)...
    ];

% Known power injections

P2 = 0.5;
P3 = -1.2;
Q3 = -0.5;

% Known theta's and voltage levels
% In this code, t = theta.
t1 = 0;
v1 = 1.02;
v2 = 1.00;

% Initialize variables
% The default values are that of a flat start.
t2 = 1;
t3 = -1;
v3 = 2;

tolerance = 1e-10;

shouldIterate = true;

iterationK = 0;

while (shouldIterate)
    
    display(strcat('Iteration number (k) =', num2str(iterationK)))
    disp(' ')
    display(strcat( ...
        'Current values of [t2 t3 v3] = [', ...
        num2str([t2 t3 v3]), ...
        ']'...
    ))
   
    % Compute the error with current iterate.
    % errors = [    Delta P2; ...
    %               Delta P3; ...
    %               Delta Q3; ]

    errors = [ P2 ...
                - v2 * v1 * abs(Y(2,1)) * cos(t2 - t1 - angle(Y(2,1))) ...
                - v2 * v2 * abs(Y(2,2)) * cos(t2 - t2 - angle(Y(2,2))) ...
                - v2 * v3 * abs(Y(2,3)) * cos(t2 - t3 - angle(Y(2,3))); ...
               ...
               P3 ...
                - v3 * v1 * abs(Y(3,1)) * cos(t3 - t1 - angle(Y(3,1))) ...
                - v3 * v2 * abs(Y(3,2)) * cos(t3 - t2 - angle(Y(3,2))) ...
                - v3 * v3 * abs(Y(3,3)) * cos(t3 - t3 - angle(Y(3,3))); ...
               ...
               Q3 ...
                - v3 * v1 * abs(Y(3,1)) * sin(t3 - t1 - angle(Y(3,1))) ...
                - v3 * v2 * abs(Y(3,2)) * sin(t3 - t2 - angle(Y(3,2))) ...
                - v3 * v3 * abs(Y(3,3)) * sin(t3 - t3 - angle(Y(3,3))); ...
             ];

%     errors
    disp(' ')
    display(strcat( ...
        '2-norm of errors = ', ...
        num2str(norm(errors, 2)) ...
    ))
    disp(' ')
    
%     pause
    
    if norm(errors, 2) > tolerance

        % Compute the Jacobian
        disp('Computing the Jacobian')
        J = computeJacobian_HW(t1, t2, t3, v1, v2, v3, Y);
        display(J)

        % Make sure that Jacobian is non singular
        assert(abs(det(J)) > 1e-5)
        
        % Compute the change in the variables.
        DeltaVars = - inv(J) * errors;
            % DeltaVars = [ Delta_t2; ...
            %               Delta_t3; ...
            %               Delta_v3; ]
        
        % Compute new variables
        t2 = t2 + DeltaVars(1);
        t3 = t3 + DeltaVars(2);
        v3 = v3 + DeltaVars(3);
        
        iterationK = iterationK + 1;
    else
        % Error is within tolerance
        shouldIterate = false;
    end

    if iterationK == 50
        disp('Did not converge within 50 iterations')
        break
    end
end
% Display the final iterate, regardless of whether the process
% converged within 50 iterations
display(strcat( ...
    'The last iterate [t2 t3 v3] = [', ...
    num2str([t2 t3 v3]), ...
    ']'...
))
