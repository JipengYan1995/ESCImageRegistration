void __global__ affine_transform2d(
    double*  d_out,
    double*  d_error,
    double* const d_Iref,
    double* const d_Imoving,
    double* const d_mx,
    int const num_row,int const N) {
    int const ind = blockDim.x * blockIdx.x + threadIdx.x;
    int indx,indy,ind2;
    //Centre of the transformation is the center of the image
    double locx,locy,locx_original, locy_original;//locx,locy are the location after transform
    int num_col=N/num_row;
    double center_x=num_col/2.0;//rotation centre
    double center_y=num_row/2.0;

    if (ind < N) {
        indx=ind/num_row;
        indy=ind%num_row;
        //x=[MI(1,1),MI(2,1),MI(1,2),MI(2,2),MI(1,3),MI(2,3)]'
        locx=d_mx[0]*(indx-center_x)+ d_mx[2]*(indy-center_y)+ d_mx[4]+center_x;
        locy= d_mx[1]*(indx-center_x)+ d_mx[3]*(indy-center_y)+ d_mx[5]+center_y;
        locx_original = locx;
        locy_original = locy;
        //after moving, if out of range project to its nearest neighbour, otherwise conduct bilinear interpolation

        if(locx_original<0 || locx_original>N/num_row-1 || locy_original<0 || locy_original>num_row-1)
        {
            d_out[ind]=9999; // to label which pixel is 'out of boundary' in Imoving
            d_error[ind]=0;
        }
        else
        {
            double tmp1,tmp2,tmp3,tmp4,tmpx,tmpy,tmp;
            if(locx<0)locx=0.5;
            if(locx>N/num_row-1)locx=N/num_row-1.0-0.5;
            if(locy<0)locy=0.5;
            if(locy>num_row-1)locy=num_row-1.0-0.5;

            //fetch intensity of the left up grid
            ind2=floor(locx)*num_row+floor(locy);
            tmp1=d_Imoving[ind2];
            //fetch intensity of the right up grid
            ind2=(floor(locx)+1)*num_row+floor(locy);
            tmp2=d_Imoving[ind2];
            //bilinear interpolation
            tmpx=tmp1*(floor(locx)+1-locx)+tmp2*(locx-floor(locx));
            //fetch intensity of the right down grid
            ind2=(floor(locx)+1)*num_row+floor(locy)+1;
            tmp4=d_Imoving[ind2];
            //fetch intensity of the left down grid
            ind2=floor(locx)*num_row+floor(locy)+1;
            tmp3=d_Imoving[ind2];
            //bilinear interpolation
            tmpy=tmp3*(floor(locx)+1-locx)+tmp4*(locx-floor(locx));
            tmp=(floor(locy)+1-locy)*tmpx+(locy-floor(locy))*tmpy;
            d_out[ind]=tmp;
            d_error[ind]=powf(tmp-d_Iref[ind],2);
        }

    }
}