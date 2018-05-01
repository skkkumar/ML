function visualize(W,rgb),

if (~exist('rgb'))
    rgb = 1;
end

% how many pixels for borders?
borders = 1;
%fprintf('Visualizing patches\n');
[ndim,nunits]=size(W);
if (rgb==3)
    ndim = ndim/3;
    if (mod(size(W,1),3))
        fprintf('Error in visualize(): rgb==3 but dimensionality not divisible by 3');
    end
end
npix = floor(sqrt(ndim)+0.999);
npix2 = floor(sqrt(nunits)+0.999);
minW=min(W(:));
maxW=max(W(:));
for j=1:rgb,
    bigpic{j} = (minW+maxW)/2*ones(((npix+borders)*npix2+borders));
    if (nunits/npix2<=npix2-1),
        if exist('horiz'),
            bigpic{j} = bigpic{j}(:,1:(npix+borders)*(npix2-1)+borders);
        else
            bigpic{j} = bigpic{j}(1:(npix+borders)*(npix2-1)+borders,:);
        end;
    end;
    for i=1:nunits;
        bigpic{j}(floor((i-1)/npix2)*(npix+borders)+borders+1:floor((i-1)/npix2)*(npix+borders)+borders+npix,...
            mod(i-1,npix2)*(npix+borders)+borders+1:mod(i-1,npix2)*(npix+borders)+borders+npix)...
               = reshape(W((1:ndim)+(j-1)*ndim,i),npix,npix)';
    end;
end
if (rgb==1)
    imagesc(-bigpic{1});
    colormap(gray);
else
    bigpic3 = repmat(bigpic{1}, [1 1 3]);
    bigpic3(:,:,2) = bigpic{2};
    bigpic3(:,:,3) = bigpic{3};
    bigpic3 = (bigpic3 - minW) / (maxW-minW);
    image(bigpic3);
end
axis off;
axis equal;
%fprintf('done.\n');

