#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>
#include <string>


void convertBGRtoOpponent( const cv::Mat& bgrImage, std::vector<cv::Mat>& opponentChannels )
{  
    opponentChannels.resize( 3 );
    opponentChannels[0] = cv::Mat(bgrImage.size(), CV_64FC1);  
    opponentChannels[1] = cv::Mat(bgrImage.size(), CV_64FC1);  
    opponentChannels[2] = cv::Mat(bgrImage.size(), CV_64FC1);  


    std::vector<cv::Mat> splitChannels;
    cv::split(bgrImage, splitChannels);

    cv::Mat O1;
    cv::add(splitChannels[1], -1*splitChannels[2], O1);
    cv::Mat O2;
    cv::add(splitChannels[1], splitChannels[2], O2);
    cv::Mat O3;
    cv::add(splitChannels[0], O2, O3);
    cv::add(O2, -2*splitChannels[0], O2);


    cv::divide(O1, O3, opponentChannels[0]);  
    cv::divide(O2, O3, opponentChannels[1]);  
    opponentChannels[2] = O3; 
}


double csf(double s, double amplitude1, double amplitude2, double sigma1, double sigma2, double contrast_min)
{
    if ( s <= 0)
    {
        return amplitude1*exp( -(s*s)/(2*sigma1*sigma1) );
    }
    else
    {
        return (amplitude2*exp( -(s*s)/(2*sigma2*sigma2) ) + contrast_min);
    }

}

cv::Mat generate_csf(const cv::Mat& zctr, int nlevel, bool intensity)
{

    double fCsfMax, fCsfMin;
    if (intensity)
    {
        fCsfMax = csf(nlevel - 4, 4.981624, 4.981624, 1.021035, 1.048155, 0);
        fCsfMin = csf(nlevel - 4.530974, 1, 0, 0.212226, 0.212226, 1);
    }
    else
    {
        fCsfMax = csf(nlevel - 4.724440, 3.611746, 3.611746, 1.360638, 0.796124, 0);
        fCsfMin = csf(nlevel - 5.059210, 1, 0, 0.348766,  0.348766, 1);        
    }
    
    return ( zctr * fCsfMax + fCsfMin*cv::Mat::ones(zctr.size(), CV_64FC1) );

}

cv::Mat relative_contrast(const cv::Mat& wlet_channel, int orientation, int center_size, int surround_size)
{

    if (wlet_channel.empty())
        return cv::Mat();
    cv::Mat wlet2 = cv::Mat::zeros(wlet_channel.size(), CV_64FC1);
    for ( int i = 0; i < wlet_channel.rows; ++i )
    {
        for ( int j = 0; j < wlet_channel.cols; ++j )
            wlet2.at<double>(i,j) = wlet_channel.at<double>( i, j)*wlet_channel.at<double>( i, j);
    }  

    cv::Mat var_cen;
    cv::Mat var_sur;
    cv::Mat roi;
    if ( orientation == 0 )
    {
        
        cv::Mat hc(cv::Mat::ones(center_size, 1, CV_64FC1));
        cv::Mat hs(cv::Mat::ones(2*surround_size + center_size, 1, CV_64FC1));
        cv::Mat temp(cv::Mat::zeros(center_size, 1, CV_64FC1));
        roi = hs(cv::Rect(0, surround_size, hs.cols, center_size));
        temp.copyTo(roi);

        cv::filter2D(wlet2, var_cen, -1 , hc, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
        var_cen = var_cen/std::count(hc.begin<double>(), hc.end<double>(), 1);

        cv::filter2D(wlet2, var_sur, -1 , hs, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
        var_sur = var_sur/std::count(hs.begin<double>(), hs.end<double>(), 1);

    }
    else if ( orientation == 1 )
    {
        cv::Mat hc(cv::Mat::ones(1, center_size, CV_64FC1));
        cv::Mat hs(cv::Mat::ones(1, 2*surround_size + center_size, CV_64FC1));
        cv::Mat temp(cv::Mat::zeros(1, center_size, CV_64FC1));
        roi = hs(cv::Rect(surround_size, 0, center_size, hs.rows));
        temp.copyTo(roi);

        cv::filter2D(wlet2, var_cen, -1 , hc, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
        var_cen = var_cen/std::count(hc.begin<double>(), hc.end<double>(), 1);

        cv::filter2D(wlet2, var_sur, -1 , hs, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
        var_sur = var_sur/std::count(hs.begin<double>(), hs.end<double>(), 1);

    }
    else
    {
        cv::Mat hc(cv::Mat::zeros(center_size, center_size, CV_64FC1));
        for ( int i = 0; i < hc.rows; ++i )
        {
            for ( int j = 0; j < hc.cols; ++j )
            {
                if ( (i + j) == (center_size - 1) || (i == j) )
                {
                    hc.at<double>(i, j) = 1;

                }
            }
        }  

        cv::Mat hs(cv::Mat::zeros(2*surround_size + center_size, 2*surround_size + center_size, CV_64FC1));
        for ( int i = 0; i < hs.rows; ++i )
        {
            for ( int j = 0; j < hs.cols; ++j )
            {
                if ( (i + j) == (2*surround_size + center_size - 1) || (i == j))
                {
                    hs.at<double>(i, j) = 1;

                }
            }
        }  

        cv::Mat temp(cv::Mat::zeros(center_size, center_size, CV_64FC1));
        temp.copyTo(hs(cv::Rect(surround_size, surround_size, center_size, center_size)));

        cv::filter2D(wlet2, var_cen, -1 , hc, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
        var_cen = var_cen/std::count(hc.begin<double>(), hc.end<double>(), 1);

        cv::filter2D(wlet2, var_sur, -1 , hs, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
        var_sur = var_sur/std::count(hs.begin<double>(), hs.end<double>(), 1);

    }

    cv::Mat zctr(cv::Mat::zeros(wlet_channel.size(), CV_64FC1));
    for ( int i = 0; i < zctr.rows; ++i )
    {
        for ( int j = 0; j < zctr.cols; ++j )
        {

            zctr.at<double>(i,j) = var_cen.at<double>(i,j)/(var_sur.at<double>(i,j) + 1e-6);
            zctr.at<double>(i,j) = zctr.at<double>(i,j) * zctr.at<double>(i,j)/
                                 ( zctr.at<double>(i,j) * zctr.at<double>(i,j) + 1 ); 
        }
    }

    return zctr;

}


cv::Mat idwt_gabor(const std::vector<cv::Mat>& wlet_channels, const cv::Mat& max_level_app, 
                   const std::vector<double>& filter, cv::Size src_size)
{
    int nlevel = wlet_channels.size()/3;
    int len = filter.size();
    cv::Mat kernel1 = cv::Mat::ones(1, len, CV_64FC1);
    cv::Mat kernel2 = cv::Mat::ones(len, 1, CV_64FC1);
    memcpy(kernel1.data, filter.data(), len*sizeof(double));
    memcpy(kernel2.data, filter.data(), len*sizeof(double));
    cv::Mat image = max_level_app;
    cv::Mat up_image;

    for (int k = nlevel - 1; k > -1; --k)
    {

        // upsample
        up_image = ( cv::Mat::zeros(image.rows*2, image.cols*2, CV_64FC1) );
        for ( int i = 0; i < image.rows; ++i )
        {
            for ( int j = 0; j < image.cols; ++j )
                up_image.at<double>(2*i, 2*j) = image.at<double>(i, j);
        }  

        cv::Mat temp;
        cv::filter2D(up_image, temp, -1 , kernel2, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
        temp = 2*temp;

        cv::filter2D(temp, temp, -1 , kernel1, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
        temp = 2*temp;

        cv::add(temp, wlet_channels[k*3 + 0], up_image);
        cv::add(up_image, wlet_channels[k*3 + 1], up_image);
        cv::add(up_image, wlet_channels[k*3 + 2], up_image);

        image = up_image.clone();

    }
    int col_extern = (exp2(ceil((log2(src_size.width)))) - src_size.width);
    int row_extern  = (exp2(ceil((log2(src_size.height)))) - src_size.height);

    return image(cv::Rect(col_extern/2, row_extern/2, src_size.width, src_size.height));

}

void dwt_gabor(const cv::Mat& opp_img, const std::vector<double>& filter, std::vector<cv::Mat>& wlet_channels, 
               std::vector<cv::Mat>& app_channels, int nlevel = 5)
{
    cv::Mat pad_img;
    int nearest_pow = exp2(ceil((log2(std::max(opp_img.cols, opp_img.rows)))));
    int col_extern = (exp2(ceil((log2(opp_img.cols)))) - opp_img.cols);
    int row_extern  = (exp2(ceil((log2(opp_img.rows)))) - opp_img.rows);
    int top, bottom, right, left;
    if (col_extern % 2 == 0)
    {
        right = left = col_extern/2;
  
    }
    else
    {
        left = col_extern/2;
        right = col_extern/2 + 1;
    }  
    if (row_extern % 2 == 0)
    {
        top = bottom = row_extern/2;
  
    }
    else
    {
        top = row_extern/2;
        bottom = row_extern/2 + 1;
    }
    copyMakeBorder( opp_img, pad_img, top, bottom, left, right, cv::BORDER_REFLECT);

    
    int len = filter.size();
    cv::Mat kernel1 = cv::Mat::ones(1, len, CV_64FC1);
    cv::Mat kernel2 = cv::Mat::ones(len, 1, CV_64FC1);
    memcpy(kernel1.data, filter.data(), len*sizeof(double));
    memcpy(kernel2.data, filter.data(), len*sizeof(double));

    for ( int k = 0; k < nlevel; ++k )
    {

        cv::Mat HF;
        filter2D(pad_img, HF, -1 , kernel1, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );

        // downsample
        cv::Mat temp_HF(cv::Mat::zeros(HF.size(), CV_64FC1));
        for ( int i = 0; i < temp_HF.rows; ++i )
        {
            for ( int j = 0; j < temp_HF.cols; j = j + 2 )
                temp_HF.at<double>(i, j) = HF.at<double>(i, j);
        }        
        cv::Mat temp_HF1;
        cv::filter2D(temp_HF, temp_HF1, -1 , kernel1, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );

        cv::Mat GF;
        cv::add(pad_img, -2*temp_HF1, GF);

        cv::Mat HHF;
        cv::filter2D(HF, HHF, -1 , kernel2, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );

        temp_HF = cv::Mat::zeros(HF.size(), CV_64FC1);
        for ( int i = 0; i < temp_HF.rows; i = i + 2 )
        {
            for ( int j = 0; j < temp_HF.cols; ++j )
                temp_HF.at<double>(i, j) = HHF.at<double>(i, j);
        }
        cv::filter2D(temp_HF, temp_HF1, -1 , kernel2, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );


        cv::Mat GHF;
        cv::add(HF, -2*temp_HF1, GHF);
        cv::filter2D(GF, temp_HF1, -1 , kernel2, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );

        temp_HF = cv::Mat::zeros(temp_HF1.size(), CV_64FC1);
        for ( int i = 0; i < temp_HF.rows; i = i + 2 )
        {
            for ( int j = 0; j < temp_HF.cols; ++j )
                temp_HF.at<double>(i, j) = temp_HF1.at<double>(i, j);
        }  
        cv::Mat HGF;
        cv::filter2D(temp_HF, HGF, -1 , kernel2, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
        HGF = 2 * HGF;

        // save the horizon/vertical wavelet plane
        wlet_channels.push_back(HGF);
        wlet_channels.push_back(GHF);

        cv::Mat HHF_dws( cv::Mat::zeros(HHF.rows/2, HHF.cols/2, CV_64FC1) );
        for ( int i = 0; i < HHF_dws.rows; ++i )
        {
            for ( int j = 0; j < HHF_dws.cols; ++j )
                HHF_dws.at<double>(i, j) = HHF.at<double>(2 * i, 2 * j);
        }  
        // save the residual plane
        app_channels.push_back(HHF_dws);

        cv::Mat HHF_ups( cv::Mat::zeros(HHF.size(), CV_64FC1) );
        for ( int i = 0; i < HHF_dws.rows; ++i  )
        {
            for ( int j = 0; j < HHF_dws.cols; ++j )
                HHF_ups.at<double>(2*i,2*j) = HHF_dws.at<double>(i, j);
        }  

        cv::filter2D(HHF_ups, temp_HF, -1 , kernel2, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
        temp_HF = 2 * temp_HF;
        cv::filter2D(temp_HF, temp_HF, -1 , kernel1, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
        temp_HF = 2 * temp_HF;

        cv::Mat DF;
        cv::add(pad_img, -1*temp_HF, DF);
        cv::add(-1*HGF, DF, DF);
        cv::add(-1*GHF, DF, DF);

        // save the diagonal plane
        wlet_channels.push_back(DF);

        // downsample
        HHF_dws = ( cv::Mat::zeros(HHF.rows/2, HHF.cols/2, CV_64FC1) );
        for ( int i = 0; i < HHF_dws.rows; ++i )
        {
            for ( int j = 0; j < HHF_dws.cols; ++j )
                HHF_dws.at<double>(i, j) = HHF_ups.at<double>(2 * i, 2 * j);
        }  

        pad_img = HHF_dws;

    }

}


cv::Mat idwt2(const std::vector<cv::Mat>& wlet_channels, const cv::Mat& max_level_app, 
              const std::vector<std::vector<double>>& filter, cv::Size src_size)
{
    int nlevel = wlet_channels.size()/3;
    int len1 = filter.size();
    int len2 = filter[0].size();

    cv::Mat image = max_level_app;
    cv::Mat up_image;
    std::vector<double> _1filter;
    for (auto ft: filter)
    {
        for(auto f: ft)
        {

            _1filter.push_back(f);
            
        }
    }
    cv::Mat kernel1 = cv::Mat::zeros(len1, len2, CV_64FC1);
    memcpy(kernel1.data, _1filter.data(), len1*len2*sizeof(double));
    
    cv::Mat kernel2 = cv::Mat::zeros(len2, len1, CV_64FC1);
    //memcpy(kernel2.data, _1filter.data(), len1*len2*sizeof(double));
    // rotate the filter
    cv::flip(kernel1, kernel2, -1);

    for (int k = nlevel - 1; k > -1; --k)
    {

        up_image = ( cv::Mat::zeros(image.rows*2, image.cols*2, CV_64FC1) );
        for ( int i = 0; i < image.rows; ++i )
        {
            for ( int j = 0; j < image.cols; ++j )
                up_image.at<double>(2*i, 2*j) = image.at<double>(i, j);
        }  

        cv::Mat temp;
        cv::filter2D(up_image, temp, -1 , kernel2, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
        temp = 2*temp;

        cv::filter2D(temp, temp, -1 , kernel1, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
        temp = 2*temp;

        cv::add(temp, wlet_channels[k*3 + 0], up_image);
        cv::add(up_image, wlet_channels[k*3 + 1], up_image);
        cv::add(up_image, wlet_channels[k*3 + 2], up_image);

        image = up_image.clone();

    }

    int col_extern = (exp2(ceil((log2(src_size.width)))) - src_size.width);
    int row_extern  = (exp2(ceil((log2(src_size.height)))) - src_size.height);
    return image(cv::Rect(col_extern/2, row_extern/2, src_size.width, src_size.height));

}


void dwt2(const cv::Mat& opp_img, const std::vector<std::vector<std::vector<double>>>& filters, 
          std::vector<cv::Mat>& wlet_channels, std::vector<cv::Mat>& app_channels, int nlevel = 5)
{

    cv::Mat pad_img;
    int nearest_pow = exp2(ceil((log2(std::max(opp_img.cols, opp_img.rows)))));
    int col_extern = (exp2(ceil((log2(opp_img.cols)))) - opp_img.cols);
    int row_extern  = (exp2(ceil((log2(opp_img.rows)))) - opp_img.rows);
    int top, bottom, right, left;
    if (col_extern % 2 == 0)
    {
        right = left = col_extern/2;
  
    }
    else
    {
        left = col_extern/2;
        right = col_extern/2 + 1;
    }  
    if (row_extern % 2 == 0)
    {
        top = bottom = row_extern/2;
  
    }
    else
    {
        top = row_extern/2;
        bottom = row_extern/2 + 1;
    }
    copyMakeBorder( opp_img, pad_img, top, bottom, left, right, cv::BORDER_REFLECT);


    std::vector<double> _4filter;
    for (auto filter: filters)
    {
        for(auto ft: filter)
        {
            for (auto f: ft)
            {
                _4filter.push_back(f);
            }
        }
    }

    int len1 = filters[0].size();
    int len2 = filters[0][0].size();
    cv::Mat kernel0 = cv::Mat::zeros(len1, len2, CV_64FC1);
    memcpy(kernel0.data, _4filter.data(), len1*len2*sizeof(double));
    cv::Mat kernel1 = cv::Mat::zeros(len1, len2, CV_64FC1);
    memcpy(kernel1.data, _4filter.data() + len1*len2, len1*len2*sizeof(double));
    cv::Mat kernel2 = cv::Mat::zeros(len1, len2, CV_64FC1);
    memcpy(kernel2.data, _4filter.data() + 2*len1*len2, len1*len2*sizeof(double));
    cv::Mat kernel3 = cv::Mat::zeros(len1, len2, CV_64FC1);
    memcpy(kernel3.data, _4filter.data() + 3*len1*len2, len1*len2*sizeof(double));


    for ( int k = 0; k < nlevel; ++k )
    {

        cv::Mat filter_a;
        filter2D(pad_img, filter_a, -1 , kernel0, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
        cv::Mat filter_h;
        filter2D(pad_img, filter_h, -1 , kernel1, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
        cv::Mat filter_v;
        filter2D(pad_img, filter_v, -1 , kernel2, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );
        cv::Mat filter_d;
        filter2D(pad_img, filter_d, -1 , kernel3, cv::Point( -1, -1 ), 0, cv::BORDER_REFLECT );

        // save the wavelet planes
        wlet_channels.push_back(filter_h);
        wlet_channels.push_back(filter_v);
        wlet_channels.push_back(filter_d);
        
        cv::Mat HHF_dws( cv::Mat::zeros(filter_a.rows/2, filter_a.cols/2, CV_64FC1) );
        for ( int i = 0; i < HHF_dws.rows; ++i )
        {
            for ( int j = 0; j < HHF_dws.cols; ++j )
                HHF_dws.at<double>(i, j) = filter_a.at<double>(2 * i, 2 * j);
        }  

        app_channels.push_back(HHF_dws);
        pad_img = HHF_dws;

    }

}


cv::Mat compute_saliency_map(const cv::Mat& opp_img, const std::vector<double>& filter, 
                             int nlevel, int center_size, int surround_size, bool intensity = false)
{
    if (opp_img.empty())
       return cv::Mat();
    std::vector<cv::Mat> wlet_channels; 
    std::vector<cv::Mat> app_channels;
    dwt_gabor(opp_img, filter, wlet_channels, app_channels, nlevel);

    std::vector<cv::Mat> wp;
    for (int i = 0; i < nlevel; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            
            cv::Mat zctr = relative_contrast(wlet_channels[i*3 + j], j, center_size, surround_size);
            // decompose level +1 
            cv::Mat alpha = generate_csf(zctr, i + 1, intensity);
            wp.push_back(alpha);

        }
    }

    cv::Mat max_level_app = cv::Mat::zeros(app_channels[nlevel - 1].size(), CV_64FC1);
    cv::Mat rec_map = idwt_gabor(wp, max_level_app, filter, opp_img.size());
    cv::Scalar s = cv::sum(rec_map);
    rec_map = rec_map/(s[0] + s[1] + s[2] + s[3]);

    return rec_map;  

}


cv::Mat compute_saliency_map(const cv::Mat& opp_img, const std::vector<std::vector<std::vector<double>>>& filters, 
        int nlevel, int center_size, int surround_size, bool intensity = false)
{
    if (opp_img.empty())
       return cv::Mat();

    std::vector<cv::Mat> wlet_channels; 
    std::vector<cv::Mat> app_channels;
    if (filters.size() == 1)
    dwt_gabor(opp_img, filters[0][0], wlet_channels, app_channels, nlevel);
    if (filters.size() == 4)
    {
        dwt2(opp_img, filters, wlet_channels, app_channels, nlevel);
    }

    std::vector<cv::Mat> wp;
    for (int i = 0; i < nlevel; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            
            cv::Mat zctr = relative_contrast(wlet_channels[i*3 + j], j, center_size, surround_size);
            // decompose level +1 
            cv::Mat alpha = generate_csf(zctr, i + 1, intensity);
            wp.push_back(alpha);
        }
    }

    cv::Mat max_level_app = cv::Mat::zeros(app_channels[nlevel - 1].size(), CV_64FC1);
    cv::Mat rec_map;
    if (filters.size() == 1)
    rec_map = idwt_gabor(wp, max_level_app, filters[0][0], opp_img.size());
    if (filters.size() == 4)
    {
        rec_map = idwt2(wp, max_level_app, filters[0], opp_img.size());
    }    

    cv::Scalar s = cv::sum(rec_map);
    rec_map = rec_map/(s[0] + s[1] + s[2] + s[3]);
    return rec_map;  

}


int main(int argc, char* argv[])
{

    std::vector<double> gabor_filter{1./16, 1./4, 3./8, 1./4, 1./16};

    std::vector<std::vector<std::vector<double>>> haar_filters{
        std::vector<std::vector<double>>{
            std::vector<double>{1/4., 1/4.}, std::vector<double>{1/4., 1/4.}},

        std::vector<std::vector<double>>{
            std::vector<double>{-1/4., -1/4.}, std::vector<double>{1/4., 1/4.}},

        std::vector<std::vector<double>>{
            std::vector<double>{1/4., -1/4.}, std::vector<double>{1/4., -1/4.}},
        std::vector<std::vector<double>>{ 
            std::vector<double>{-1/4., 1/4.}, std::vector<double>{1/4., -1/4.}}  };


    std::vector<std::vector<std::vector<double>>> nonwavelet_filters{
        std::vector<std::vector<double>>{
            std::vector<double>{-1/8., 1/8., 1/8., 1/8.}, std::vector<double>{1/8., 1/8., -1/8., 1/8.},
            std::vector<double>{1/8., -1/8., 1/8., 1/8.}, std::vector<double>{1/8., 1/8., 1/8., -1/8.}},

        std::vector<std::vector<double>>{
            std::vector<double>{-1/8., 1/8., -1/8., -1/8.}, std::vector<double>{1/8., 1/8., 1/8., -1/8.},
            std::vector<double>{1/8., -1/8., -1/8., -1/8.}, std::vector<double>{1/8., 1/8., -1/8., 1/8.}},

        std::vector<std::vector<double>>{
            std::vector<double>{1/8., -1/8., -1/8., -1/8.}, std::vector<double>{-1/8., -1/8., 1/8., -1/8.},
            std::vector<double>{1/8., -1/8., 1/8., 1/8.}, std::vector<double>{1/8., 1/8., 1/8., -1/8.}},
        std::vector<std::vector<double>>{ 
            std::vector<double>{-1/8., 1/8., -1/8., -1/8.}, std::vector<double>{1/8., 1/8., 1/8., -1/8.},
            std::vector<double>{-1/8., 1/8., 1/8., 1/8.}, std::vector<double>{-1/8., -1/8., 1/8., -1/8.}}  };
    
   

    cv::Mat img = cv::imread(argv[1]);
    if (img.channels() == 3)
    {
        cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    }
    cv::imshow("src", img);
    cv::waitKey(0);
    img.convertTo(img, CV_64FC1);

    std::vector<cv::Mat> opponentChannels;
    if( img.channels() == 1 )
    {
       opponentChannels.push_back(img);
    }
    else
    {
       convertBGRtoOpponent( img, opponentChannels );

    }
    int nlevel = std::min(7.0, std::log(std::min(img.cols, img.rows))/std::log(2));
    std::vector<cv::Mat> maps;
    for (auto opp_img: opponentChannels)
    {
        
        if (std::string(argv[2]) == "gabor")
        {
            maps.push_back(compute_saliency_map(opp_img, gabor_filter, nlevel, 13, 26, false));
        }
        else if (std::string(argv[2]) == "haar")
        {
            maps.push_back(compute_saliency_map(opp_img, haar_filters, nlevel, 13, 26, false));
        }
        else if (std::string(argv[2]) == "non")
        {
            maps.push_back(compute_saliency_map(opp_img, nonwavelet_filters, nlevel, 13, 26, false));
        }
        else
        {

        }
        
    }

    cv::Mat smap( cv::Mat::zeros(maps[0].rows, maps[0].cols, CV_64FC1) );
    double smap_min = 10000;
    double smap_max = 0;
    for ( int i = 0; i < maps[0].rows; ++i )
    {
        for ( int j = 0; j < maps[0].cols; ++j )
        {
            for (auto m: maps)
            {
                smap.at<double>(i,j) = smap.at<double>(i,j) + m.at<double>(i,j)*m.at<double>(i,j);
            }
            
            if (smap.at<double>(i,j) < smap_min)
                smap_min = smap.at<double>(i,j);
            if (smap.at<double>(i,j) > smap_max)
                smap_max = smap.at<double>(i,j);            
            
        }
    } 
    std::cout << "min/max value: " << smap_min << ", " << smap_max << std::endl;

    cv::Mat smap1;
    cv::add(smap, -1*smap_min*cv::Mat::ones(smap.size(), CV_64FC1), smap1);
    smap1 = smap1/(smap_max - smap_min);
    cv::normalize(smap1, smap1, 1, 0, cv::NORM_MINMAX);
    smap1.convertTo(smap1, CV_8UC1, 255, 0);
    cv::imshow("smap1", smap1);
    cv::waitKey(0);  

    cv::Mat sum, sqsum;
	cv::integral(smap, sum, sqsum);
    int wh = 50;
    int ww = 40;
    cv::Mat sum_map = cv::Mat::zeros(smap.rows - wh, smap.cols - ww, CV_64FC1);
    for(int i = 0; i < sum_map.cols; ++i)
    {
        for (int j = 0; j < sum_map.rows; ++j)
        {
            sum_map.at<double>(j, i) = sum.at<double>(j,i) + sum.at<double>(j + wh, i + ww) - 
                      sum.at<double>(j + wh, i) - sum.at<double>(j, i + ww);
        }
    }

    cv::normalize(smap, smap, 1, 0, cv::NORM_MINMAX);
    smap.convertTo(smap, CV_8UC1, 255, 0);
    cv::imwrite("smap.png", smap);
    cv::imshow("smap", smap);
    cv::waitKey(0);

    cv::normalize(sum_map, sum_map, 1, 0, cv::NORM_MINMAX);
    sum_map.convertTo(sum_map, CV_8UC1, 255, 0);
    cv::imwrite("sum_map.png", sum_map);
    cv::imshow("sum_map", sum_map);
    cv::waitKey(0);

    return 0;

}