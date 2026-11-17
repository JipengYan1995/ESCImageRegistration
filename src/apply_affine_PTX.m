function [I_out] = apply_affine_PTX(I_in,x_out, x_scale, affine_kernels)


image_dimension=ndims(I_in);
x_out=double(x_out);
x_scale=double(x_scale);
I_in=double(I_in);
d_mx = gpuArray(zeros(image_dimension*(image_dimension+1),1,'double'));
d_x = gpuArray(x_out);
d_scale = gpuArray(x_scale);
d_out = gpuArray(zeros(size(I_in(:)),'double'));
d_Imoving = gpuArray(I_in(:));
d_array_size = gpuArray(size(I_in));
N = gpuArray(numel(I_in));
square_error=double(zeros(size(d_Imoving),'gpuArray')); 


d_mx = feval(affine_kernels{1}, d_mx, d_x, d_scale); % translate motion array into tranformation matrix

[d_out,~]= feval(affine_kernels{2}, d_out,square_error,d_Imoving, d_Imoving, d_mx, d_array_size(1), N); 

I_out = gather(d_out);
I_out = reshape(I_out,size(I_in));
I_out(I_out==9999)=0;
end