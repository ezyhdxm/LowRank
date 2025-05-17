function Z = solve_dnn_with_sdpnal(C, m)
    n = size(C, 1);
    K = n - m;

    % Define block: symmetric matrix block of size n
    blk = cell(1,1); 
    blk{1,1} = 's'; 
    blk{1,2} = n;

    % Cost matrix
    Ccell = cell(1,1);
    Ccell{1} = C;

    % Build A and b
    % 1. Diagonal constraints: Z_ii <= 1 ⇒ -Z_ii ≥ -1
    A = cell(1,1); At = [];
    cnt = 0;

    % Constraint 1: -Z_ii ≤ -1  (i.e., Z_ii ≤ 1)
    for i = 1:n
        tmp = sparse(n,n); tmp(i,i) = -1;
        cnt = cnt + 1;
        At(:,cnt) = svec(blk, tmp);  % vectorize
    end
    b_diag = -ones(n,1);

    % Constraint 2: trace(Z) = K ⇒ sum Z_ii = K
    tmp = sparse(n,n); tmp(1:n+1:end) = 1;  % diagonal
    cnt = cnt + 1;
    At(:,cnt) = svec(blk, tmp);
    b_trace = K;

    % Constraint 3: sum of all entries = K^2
    tmp = sparse(n,n); tmp(:) = 1;
    cnt = cnt + 1;
    At(:,cnt) = svec(blk, tmp);
    b_sum = K^2;

    A{1} = At;
    b = [b_diag; b_trace; b_sum];

    % Run SDPNAL+
    OPTIONS.tol = 1e-6;
    OPTIONS.printlevel = 1;
    [obj,X,y,Z,info] = sdpnalplus(blk, A, Ccell, b, OPTIONS);

    % Extract solution
    Z = smat(blk, X{1});
end
