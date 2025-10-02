function [res_norm, varargout] = res_fac(A, precs, varargin)
%RES_FAC  Residual factorization of solution factors of Lyapunov equations.
%
%   SYNTAX:  
%
%   1.     [RES_NORM, L_PLUS, L_MINUS] = RES_FAC(A, PRECS, L, Z)
%   factorizes the residual Res(Z) = A*Z*Z^T + Z*Z^T*A^T + L*L^T into 
%   positive semidefinite- and negative semidefinite- parts as 
%   Res(Z) = L_PLUS*L_PLUS^T - L_MINUS*L_MINUS^T in precision PRECS.
% 
%   2.   [RES_NORM, L_PLUS, L_MINUS] = RES_FAC(A, PRECS, L, Z, RES_DCMTOL)
%   additionally specifies the tolerance in the residual factorization to 
%   RES_DCMTOL.
%
%   3.     [RES_NORM, L_DELTA, S_DELTA] = RES_FAC(A, PRECS, L, S, Z, Y)
%   recasts the residual Res(Z) = A*Z*Y*Z^T + Z*Y*Z^T*A^T + L*S*L^T as 
%   L_DELTA*S_DELTA*L_DELTA^T in precision PRECS, with truncation on small 
%   eigenvalues.
%
%   4.     [RES_NORM, L_DELTA, S_DELTA] = 
%                               RES_FAC(A, PRECS, L, S, Z, Y, RES_DCMTOL)
%   additionally specifies the tolerance in the residual factorization to 
%   RES_DCMTOL.
%
%   RES_DCMTOL = 1e-4 by default;
%   RES_NORM retuens an estimate of the Frobenius norm of the residual.

narginchk(4, 7);
res_dectol_default = 1e-4;

if nargin == 4 
    type = sprintf('chol');
    res_dcmtol = res_dectol_default;
    L = varargin{1};
    Z = varargin{2};
elseif nargin == 5
    type = sprintf('chol');
    L = varargin{1};
    Z = varargin{2};
    res_dcmtol = varargin{3};
elseif nargin == 6
    type = sprintf('ldlt');
    res_dcmtol = res_dectol_default;
    L = varargin{1};
    S = varargin{2};
    Z = varargin{3};
    Y = varargin{4};
else
    type = sprintf('ldlt');
    L = varargin{1};
    S = varargin{2};
    Z = varargin{3};
    Y = varargin{4};
    res_dcmtol = varargin{5};
end

[n, m] = size(L);
p = size(Z, 2);
switch precs
    case 0
        precs = sprintf('half');
        Z = chop(Z);
        A = chop(A);
        L = chop(L);
    case 1
        precs = sprintf('single');
        Z = single(Z);
        A = single(A);
        L = single(L);
     case 2
        precs = sprintf('double');
        Z = double(Z);
        A = double(A);
        L = double(L);
     case 4
        precs = sprintf('quad');
        mp.Digits(34);
        Z = mp(Z);
        A = mp(A);
        L = mp(L);
    otherwise
        disp('Error: residual decomposition precision invalid.')
end
  
switch type
    case {'chol'}
        F = [Z, A*Z, L];  
        switch precs
            case {'half', 'single', 'double'}
                [U, T] = qr(F, 'econ');    % T: (2p+m)-by-(2p+m)
                % form the permutation matrix P and the small core matrix B
                if strcmp(precs,'single') 
                    Ip = eye(p, precs);
                    Im = eye(m, precs);
                    Op = zeros(p, precs);
                    Opm = zeros(p, m, precs);
                else
                    Ip = eye(p, 'double');
                    Im = eye(m, 'double');
                    Op = zeros(p, 'double');
                    Opm = zeros(p, m, 'double');
                end     
                Omp = Opm';
                P = [Op Ip Opm; Ip Op Opm; Omp Omp Im];
                H = T * P * T'; 
            case 'quad'
                if n>2*p+m
                    [U, T] = qr(F);    % T: (2p+m)-by-(2p+m)
                    U = U(:,1:(2*p+m));
                    T = T(1:(2*p+m),1:(2*p+m));
                    % form the permutation matrix P and the small core matrix S
                    Ip = eye(p,'mp');
                    Im = eye(m,'mp');
                    Op = zeros(p,'mp');
                    Opm = zeros(p,m,'mp');
                    Omp = Opm';
                    P = [Op Ip Opm; Ip Op Opm; Omp Omp Im];
                    H = T * P * T'; 
                else
                    H = Z*(Z'*A');
                    H = H + H' + L*L';
                end
        end

        % truncation and spectrum splitting 
        [Q, vec_lambda] = eig(H, 'vector');
        decomtol = res_dcmtol * max(abs(vec_lambda)); 
        res_norm = sqrt(vec_lambda'*vec_lambda);

        % if isinf(res_norm) keyboard; end

        plus_index = vec_lambda>=decomtol;
        minus_index = vec_lambda<=-decomtol;
        
        % plus_index
        % minus_index
        % if all(~plus_index) || all(~minus_index) keyboard; end

        vec_lambda_plus = vec_lambda(plus_index);
        vec_lambda_minus = vec_lambda(minus_index);

        Q_plus = Q(:, plus_index);
        Q_minus = Q(:, minus_index);

        if strcmp(precs,'quad') 
            if n>2*p+m
                L_plus = U * ( Q_plus * diag(sqrt(vec_lambda_plus)) );
                L_minus = U * ( Q_minus * diag(sqrt(-vec_lambda_minus)) );
            else
                L_plus = Q_plus * diag(sqrt(vec_lambda_plus));
                L_minus = Q_minus * diag(sqrt(-vec_lambda_minus));
            end
            L_plus = double(L_plus);
            L_minus = double(L_minus);
        else
            L_plus = U * Q_plus * diag(sqrt(vec_lambda_plus));
            L_minus = U * Q_minus * diag(sqrt(-vec_lambda_minus));
        end
        varargout{1} = L_plus;
        varargout{2} = L_minus;
        res_norm = double(res_norm);
    case {'ldlt'}
        L_delta = [Z, A*Z, L];         
        Op = zeros(p, 'double');
        Opm = zeros(p, m, 'double');
        S_delta = [Op Y Opm; Y Op Opm; Opm' Opm' S];
        switch precs
            case 'half'
                S_delta = chop(S_delta);
            case 'single'
                S_delta = single(S_delta);
            case 'double'
                S_delta = double(S_delta);
            case 'quad'
                S_delta = mp(S_delta);
        end
        switch precs
            case {'half', 'single', 'double'}
                [U, T] = qr(L_delta, 'econ');   
                H = T * S_delta * T';
            case 'quad'
                if n>2*p+m
                    [U, T] = qr(L_delta);  
                    U = U(:,1:(2*p+m));
                    T = T(1:(2*p+m),1:(2*p+m));
                    H = T * S_delta * T';
                else
                    H = L_delta * S_delta * L_delta';
                end
        end

        % truncation and spectrum splitting 
        [Q, vec_lambda] = eig(H, 'vector');
        decomtol = res_dcmtol * max(abs(vec_lambda)); 
        res_norm = sqrt(vec_lambda'*vec_lambda);        
        index_keep = abs(vec_lambda) >= decomtol;
        vec_lambda_large = vec_lambda(index_keep);
        Q_keep = Q(:, index_keep);

        if strcmp(precs,'quad') 
            if n>2*p+m
                S_delta = diag(vec_lambda_large);
                L_delta = U * Q_keep;
            else
                S_delta = diag(vec_lambda_large);
                L_delta = Q_keep;
            end
            S_delta = double(S_delta);
            L_delta = double(L_delta);
        else
            S_delta = diag(vec_lambda_large);
            L_delta = U * Q_keep;
        end
        varargout{1} = L_delta;
        varargout{2} = S_delta;
        res_norm = double(res_norm);
    otherwise
        error('Specified decomposition type not supported.');
end

end