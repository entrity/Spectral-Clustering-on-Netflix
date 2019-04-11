disp('fsds')
A = [
	1 1 1 0 0 0 0 0
	1 1 1 0 0 0 0 0
	1 1 1 0 0 0 0 0
	0 0 0 0 0 1 1 1
	0 0 0 0 0 1 1 1
	0 0 0 0 0 1 1 1
	0 0 0 1 1 0 0 0
	0 0 0 1 1 0 0 0
];
d = sum(A);
D = diag(d)

% Unnormalized L
Lu = D - A;

% ala CuttingElephants
eleD = diag(d.^(-1/2));
Le = eye(8) - eleD*A*eleD;
[vec_e,val_e]=eig(Le);
[Y,I]=sort(diag(val_e),'ascend');
disp(Y)

% Normalized L
normD = diag(d.^(-1/2));
Ln = normD*Lu*normD;
[vec_n,val_n]=eig(Ln);
[Y,I]=sort(diag(val_n),'ascend');
disp(Y)

% Random walk L
randD = diag(d.^(-1));
Lr = randD * Lu;
[vec_r,val_r]=eig(Ln);
[Y,I]=sort(diag(val_r),'ascend');
disp(Y)
