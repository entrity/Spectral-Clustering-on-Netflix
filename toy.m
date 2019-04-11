function tmp ()
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
	fprintf(1,' >>> like CuttingElephants <<<\n');
	disp(vec_e(:,I))

	% Normalized L
	normD = diag(d.^(-1/2));
	Ln = normD*Lu*normD;
	[vec_n,val_n]=eig(Ln);
	[Y,I]=sort(diag(val_n),'ascend');
	fprintf(1,' >>> Normalized <<<\n');
	disp(vec_n(:,I))

	% Random walk L
	randD = diag(d.^(-1));
	Lr = randD * Lu;
	[vec_r,val_r]=eig(Ln);
	[Y,I]=sort(diag(val_r),'ascend');
	fprintf(1,' >>> Random walk <<<\n');
	disp(vec_r(:,I))

	% Why do my eigenvectors not indicate cuts?
	% Why is none of them an equilibrium distribution?
end