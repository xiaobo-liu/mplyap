function C = matmul(A, B, prec)
% matrix multiplication in double/single/half precisions.

if isequal(prec,'double') % use double precision
        C = double(A)*double(B);
elseif isequal(prec,'single') % use single precision
        C = single(A)*single(B);
else % use half precision
        C = matmul_half(A, B);
end
end

function C = matmul_half(A, B)
% matrix multiplication in half precision
if isreal(A) && isreal(B) % both A and B are real matrices
    C = matmul_half_real(A, B);
else
    imagA = imag(A);
    realA = real(A); 
    imagB = imag(B);
    realB = real(B); 
    C = matmul_half_real(realA, realB) - matmul_half_real(imagA, imagB) + ...
        1i*(matmul_half_real(realA, imagB) + matmul_half_real(imagA, realB));
end
end

function C = matmul_half_real(A, B)
% matrix multiplication between real matrices in half precision
    [m, n] = size(A);
    p = size(B, 2);
    A = chop(A);
    B = chop(B);
    C = zeros(m, p, 'double');
    for i = 1:n
        C = chop(C + chop(A(:,i)*B(i,:)));
    end
end