function [Registered_image,affined_parameter,error_array,diverge_flag]=affine_registration_by_ESC(Imoving,Iref,affine_Moco_para,affine_kernels,ESCPara)
%% initialisation
% get data size
N = (numel(Imoving));
array_size = int32(size(Imoving));
image_dimension = length(array_size);
d_array_size=gpuArray(array_size);

% initalise variables into GPU
d_scale=gpuArray(affine_Moco_para.x_scale);
d_Iref = Iref(:);d_Iref=gpuArray(double(d_Iref));
d_Imoving = Imoving(:);
d_Imoving=double(d_Imoving);
d_Imoving=gpuArray(d_Imoving);
d_out = double(zeros(size(d_Imoving),'gpuArray')); % registered image
d_mx = gpuArray(zeros(image_dimension*(image_dimension+1),1,'double')); % 2D motion transformation matrix, size 2 x 3;
square_error=double(zeros(size(d_Imoving),'gpuArray')); % image difference - squared error
affine_Moco_para.mask=gpuArray(affine_Moco_para.mask(:));
error_array=zeros(affine_Moco_para.max_iter,1); % array to record error

% initialise ESC solver
ESCPara.theta_0= affine_Moco_para.x_init;
ESCPara.theta_cumsum=zeros(size(affine_Moco_para.x_init));
ESCPara.theta = ESCPara.theta_0;


%% iteration
diverge_flag=0;
for ii=1:affine_Moco_para.max_iter

    temp =  sin(ESCPara.omega*ii);
    ESCPara.theta =  ESCPara.theta_0 + ESCPara.theta_cumsum + ESCPara.a .* temp;

    d_mx = feval(affine_kernels{1}, d_mx, gpuArray(ESCPara.theta), d_scale); % translate motion array into tranformation matrix

    [d_out,square_error]= feval(affine_kernels{2}, d_out,square_error,d_Iref, d_Imoving, d_mx, d_array_size(1), gpuArray(N)); % calculate image difference with GPU
    check =  (d_out)~=9999 & affine_Moco_para.mask;
    if  sum(check)==0
        disp('Not converge');
        break;
    else
        temp_array=square_error(check);
        error_next =  0.5.*mean(temp_array,"all"); % update error
    end

    error_next=gather(error_next);
    
    ESCPara.theta_cumsum=ESCPara.theta_cumsum - ESCPara.K * error_next * 2/ESCPara.a .* temp; 

    if  any((abs(ESCPara.theta(1:image_dimension)).*affine_Moco_para.x_scale(1:image_dimension)-double(array_size(1:image_dimension))') >0)
        disp('Not converge');
        diverge_flag=1;
        break;
    end

    error=error_next;
    error_array(ii)=gather(error) ;
end

if  diverge_flag
    disp('Not converge');
    Registered_image=Imoving;
    affined_parameter=  ESCPara.theta_0;
else
    Registered_image=reshape(gather(d_out),array_size);
    Registered_image(Registered_image==9999)=0;
    affined_parameter=  ESCPara.theta_0 + ESCPara.theta_cumsum;

end

end