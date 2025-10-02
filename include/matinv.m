function C = matinv(A, prec)
% matrix inversion in double/single/half precisions (half-precision inv is 
% done in double and then converted to simulated half).

if isequal(prec,'double') % use double precision
    C = inv(double(A));
elseif isequal(prec,'single') % use single precision
    C = inv(single(A));
else % use half precision
    C = inv(chop(A));
end

end






