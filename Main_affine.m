clear;
reset(gpuDevice);
load(['ExampleImages.mat']);
%% define affine model parameter
affine_Moco_para.x_scale=[1; 1; 0.01; 0.001; 0.001; 0.001; 0.001];  % scale for [tranlsation X (pixels), translation Z (pixels), rotation (radians); rescale X, rescale Y, shear XZ, shear ZX]
affine_Moco_para.x_init=[0;0;0;1/affine_Moco_para.x_scale(4);1/affine_Moco_para.x_scale(5);0;0]; % initalization for [tranlsation X (pixels), translation Z (pixels), rotation (radians); rescale X, rescale Y, shear XZ, shear ZX]
affine_Moco_para.mask = ones(size(Iref,1,2),'logical');
affine_Moco_para.max_iter = 1000;

% load CUDA kernels
ImageSize=size(Iref);
threadsPerBlock=256;
ThreadBlockSize=[threadsPerBlock,1,1];
GridSize=[ceil((prod(ImageSize)+threadsPerBlock-1)/threadsPerBlock),1,1];

make_affine_mat_kernel = parallel.gpu.CUDAKernel("make_transform_matix_for_ESC_kernel.ptx","make_transform_matix_for_ESC_kernel.cu");
make_affine_mat_kernel.ThreadBlockSize = [1,1,1];
make_affine_mat_kernel.GridSize  =  [1,1,1];
affine_kernels{1}=make_affine_mat_kernel;
affine_trans_kernel = parallel.gpu.CUDAKernel("affine_transform2d_interp_only_kernel.ptx","affine_transform2d_interp_only_kernel.cu");
affine_trans_kernel.ThreadBlockSize = ThreadBlockSize;
affine_trans_kernel.GridSize  = GridSize;
affine_kernels{2}=affine_trans_kernel;

% ESC solver parameter
num_para = length(affine_Moco_para.x_init);
raw_omega =  linspace(0.9*pi,0.6*pi,num_para);
ESC_affine.omega = reshape(raw_omega, [], 1);
ESC_affine.K=1;
ESC_affine.a = 0.001;

%% Image registration
Imoving_smoothed =  imgaussfilt(Imoving,10); % smooth the image can be helpful for covergence
Iref_smoothed = imgaussfilt(Iref,10);

[Imoving_smoothed_reg,affine_parameter,error_array,diverge_flag]=affine_registration_by_ESC(Imoving_smoothed,Iref_smoothed,affine_Moco_para,affine_kernels,ESC_affine);

Imoving_reg = apply_affine_PTX(Imoving, affine_parameter, affine_Moco_para.x_scale, affine_kernels); % this can be used to correct motion in CEUS frames
%% show images
figure
subplot(2,2,1)
imagesc(Iref);
axis image
colormap gray
title('Reference Image')

subplot(2,2,2)
imagesc(Imoving);
axis image
colormap gray
title('Moving Image')

subplot(2,2,3)
imshowpair(Iref,Imoving);
axis image
title('Reference Image vs. Moving Image')

subplot(2,2,4)
imshowpair(Iref,Imoving_reg);
axis image
title('Reference Image vs. Registered Moving Image')
