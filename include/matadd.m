function C = matadd(A, B, prec)
% matrix addition in double/single/half precisions.

if isequal(prec,'double') % use double precision
    C = double(A)+double(B);
elseif isequal(prec,'single') % use single precision
    C = single(A)+single(B);
else % use half precision
    C = matadd_half(A, B);
end
end

function C = matadd_half(A, B)
% matrix addition in half precision
if isreal(A) && isreal(B) % both A and B are real matrices
    C = matadd_half_real(A, B);
else
    imagA = imag(A);
    realA = real(A); 
    imagB = imag(B);
    realB = real(B); 
    C = matadd_half_real(realA, realB) + 1i*matadd_half_real(imagA, imagB);
end
end 

function C = matadd_half_real(A, B)
% matrix addition between real matrices in half precision
    C = chop(A+chop(B));
end