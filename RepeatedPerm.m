
function p=RepeatedPerm(input,num)
ndgridInput=repmat(input',[1 num]);
p=cell(1,num);
p=ndgrid(ndgridInput);
end
