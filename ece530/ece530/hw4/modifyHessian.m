
function HTilde = modifyHessian(H)
    n = size(H, 1);

    % Problem 4.1: Write your code here to compute D
    % using your answer in part (c).
    epsilon = 1e-3; % A small value to make the values truly "greater than"
    h_ii = diag(H);
    D_ii = zeros(size(h_ii));
    off_diag = H-diag(h_ii);

    for i = 1:n
        r = sum(abs(off_diag(i,:)));
        min_eigenvalue_estimate = h_ii(i) - r;
        if min_eigenvalue_estimate < (1/2)
            D_ii(i) = 1/2 + r - h_ii(i) + epsilon;
        end
    end
    max_d_ii = max(D_ii);

    % HMMMMMMMM this worked....
    D = max_d_ii * eye(n);
    HTilde = H + D;
end

