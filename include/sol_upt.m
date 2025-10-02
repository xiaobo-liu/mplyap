function [Z, Y] = sol_upt(precs, varargin)
%SOL_UPT  Factored solution update of Lyapunov equations.
%
%   SYNTAX:  
%
%   1.         Z = SOL_UPT(PRECS, Z, Z_PLUS, Z_MINUS)
%   computes the factor Z of a positive definite part of the matrix 
%   Z*Z^T + Z_PLUS*Z_PLUS^T - Z_MINUS*Z_MINUS^T in precision PRECS.
% 
%   2.      [Z, Y] = SOL_UPT(PRECS, Z, Y, Z_DELTA, Y_DELTA)
%   computes the factors Z and Y of a positive definite part of the matrix 
%   Z*Y*Z^T + Z_DELTA*Y_DELTA*Z_DELTA^T in precision specified by PRECS. 

narginchk(4, 5);

if nargin == 4 
    type = sprintf('chol');
    Z = varargin{1};
    Z_plus = varargin{2};
    Z_minus = varargin{3};
elseif nargin == 5
    type = sprintf('ldlt');
    Z = varargin{1};
    Y = varargin{2};
    Z_delta = varargin{3};
    Y_delta = varargin{4};
else
    disp('Error: sol_upt nargin invalid.')
end

[n, p] = size(Z);
switch type
    case {'chol'}
        p_plus = size(Z_plus, 2);
        p_minus = size(Z_minus, 2);
        switch precs
            case 0
                precs = sprintf('half');
                Z = chop(Z);
                Z_plus = chop(Z_plus);
                Z_minus = chop(Z_minus);
                rel_sol_dctol = 10^(-3);
            case 1
                precs = sprintf('single');
                Z = single(Z);
                Z_plus = single(Z_plus);
                Z_minus = single(Z_minus);
                rel_sol_dctol = 10 * eps('single')/2;
            case 2
                precs = sprintf('double');
                Z = double(Z);
                Z_plus = double(Z_plus);
                Z_minus = double(Z_minus);
                rel_sol_dctol = 10 * eps('double')/2;
            case 4
                precs = sprintf('quad');
                mp.Digits(34);
                Z = mp(Z);
                Z_plus = mp(Z_plus);
                Z_minus = mp(Z_minus);
                rel_sol_dctol = 10^(-33);
            otherwise
                disp('Error: sol_upt prec. invalid.')
        end
        G = [Z, Z_plus, Z_minus];         % F: n-by-(p+p_plus+p_minus)

        switch precs
            case {'half', 'single', 'double'}
                [U, T] = qr(G, 'econ');    % T: (p+p_plus+p_minus)-by-(p+p_plus+p_minus)
                
                % form the permutation matrix P and the small core matrix S
                if strcmp(precs,'single') 
                    Ip = eye(p, precs);
                    Ip_plus = eye(p_plus, precs);
                    Ip_minus = eye(p_minus, precs);
                else
                    Ip = eye(p, 'double');
                    Ip_plus = eye(p_plus, 'double');
                    Ip_minus = eye(p_minus, 'double');
                end
                H = T * blkdiag(Ip, Ip_plus, -Ip_minus) * T';
            case 'quad'
                if n>p_plus+p_minus+p
                    [U, T] = qr(G);  
                    U = U(:,1:(p_plus+p_minus+p));
                    T = T(1:(p_plus+p_minus+p),1:(p_plus+p_minus+p));

                    % form the permutation matrix P and the small core matrix S
                    Ip = eye(p, 'mp');
                    Ip_plus = eye(p_plus, 'mp');
                    Ip_minus = eye(p_minus, 'mp');
                    H = T * blkdiag(Ip, Ip_plus, -Ip_minus) * T';
                else
                    H = Z*Z' + Z_plus*Z_plus' - Z_minus*Z_minus';
                end
        end

        % truncation and spectrum splitting 
        [Q, vec_lambda] = eig(H, 'vector');

        decomtol = rel_sol_dctol * max(abs(vec_lambda)); 

        plus_index = vec_lambda>=decomtol;
        vec_lambda_plus = vec_lambda(plus_index);
        Q_plus = Q(:, plus_index);

        if strcmp(precs,'quad') 
            if n>p_plus+p_minus+p
                Z = U * Q_plus * diag(sqrt(vec_lambda_plus));
            else
                Z = Q_plus * diag(sqrt(vec_lambda_plus));
            end
            Z = double(Z);
        else
            Z = U * Q_plus * diag(sqrt(vec_lambda_plus));
        end
    case {'ldlt'}
        switch precs
            case 0
                precs = sprintf('half');
                Z = chop(Z);
                Y = chop(Y);
                Z_delta = chop(Z_delta);
                Y_delta = chop(Y_delta);
                rel_sol_dctol = 10^(-3);
            case 1
                precs = sprintf('single');
                Z = single(Z);
                Y = single(Y);
                Z_delta = single(Z_delta);
                Y_delta = single(Y_delta);
                rel_sol_dctol = 10 * eps('single')/2;
            case 2
                precs = sprintf('double');
                Z = double(Z);
                Y = double(Y);
                Z_delta = double(Z_delta);
                Y_delta = double(Y_delta);
                rel_sol_dctol = 10 * eps('double')/2;
            case 4
                precs = sprintf('quad');
                mp.Digits(34);
                Z = mp(Z);
                Y = mp(Y);
                Z_delta = mp(Z_delta);
                Y_delta = mp(Y_delta);
                rel_sol_dctol = 10^(-33);
            otherwise
                disp('Error: Solution update precision invalid.')
        end
        Z = [Z, Z_delta];
        Y = blkdiag(Y, Y_delta);
        p_delta = size(Z_delta, 2);
        % find the nearest PSD part with truncation
        switch precs
            case {'half', 'single', 'double'}
                [U, T] = qr(Z, 'econ');   
                H = T * Y * T';
            case 'quad'
                if n>p+p_delta
                    [U, T] = qr(Z);  
                    U = U(:,1:(p+p_delta));
                    T = T(1:(p+p_delta),1:(p+p_delta));
                    H = T * Y * T';
                else
                    H = Z * Y * Z';
                end
        end
        [Q, vec_lambda] = eig(H, 'vector');
        decomtol = rel_sol_dctol * max(abs(vec_lambda)); 

        plus_index = vec_lambda>=decomtol;

        vec_lambda_plus = vec_lambda(plus_index);

        Q_plus = Q(:, plus_index);
        
        if strcmp(precs,'quad') 
            if n>p_delta+p
                Y = diag(vec_lambda_plus);
                Z = U * Q_plus;
            else
                Y = diag(vec_lambda_plus);
                Z =  Q_plus;
            end
            Y = double(Y);
            Z = double(Z);
        else
            Y = diag(vec_lambda_plus);
            Z =  U * Q_plus;
        end
    otherwise
        error('Specified decomposition type not supported.');
end
end