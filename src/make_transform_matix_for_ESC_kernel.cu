__global__ void make_transform_matix_kernel(double * d_mx, double const* __restrict__ d_x, double const* __restrict__ d_scale)
{
    double d_x_scaled[7]={0};
    for (int i=0; i<7; i++){
        d_x_scaled[i] = d_x[i]*d_scale[i];
    }
    double theta_term_cos = cosf(d_x_scaled[2]);
    double theta_term_sin = sinf(d_x_scaled[2]);
    d_mx[0] =  theta_term_cos*d_x_scaled[3]+theta_term_sin*d_x_scaled[3]*d_x_scaled[6];
    d_mx[1] = -theta_term_sin*d_x_scaled[4]+theta_term_cos*d_x_scaled[4]*d_x_scaled[6];
    d_mx[2] =  theta_term_cos*d_x_scaled[3]*d_x_scaled[5]+theta_term_sin*d_x_scaled[3];
    d_mx[3] =  theta_term_cos*d_x_scaled[4]-theta_term_sin*d_x_scaled[4]*d_x_scaled[5];
    d_mx[4] = d_x_scaled[0];
    d_mx[5] = d_x_scaled[1];
}

