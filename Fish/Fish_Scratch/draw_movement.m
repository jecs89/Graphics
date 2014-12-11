function draw_movement( iterations, population, lim1, lim2 )

% graphics_toolkit( gnuplot_binary );
matriz = load('movement.data');

x = size(matriz,1)
y = size(matriz,2)

experiments = iterations;
epochs = population;

data = zeros( epochs, y );

sufix = '.jpg';

for i=1 : experiments

	idx = (i-1)*epochs;
	
	for j=1 : epochs
		data(j,:) = matriz( idx + j, : );
		% data(1,:) = matriz( idx + 1, : );
		% data(2,:) = matriz( idx + 2, : );
		% data(3,:) = matriz( idx + 3, : );
	end
		
	% figure;
	h = plot( data(:,1), data(:,2), 'b*','LineWidth', 2, 'MarkerSize', 3);
	grid on;
	axis( [lim1, lim2, lim1, lim2 ]);	

	num = int2str( i );
	name = strcat( num , sufix );
	saveas( h, name );
end

% data = data / experiments ;

% figure ;
% grid on;
% plot( data(:,1), data(:,2), 'b-','LineWidth', 2, 'MarkerSize', 7);
% hold on;

% plot( data(:,1), data(:,3), 'r-','LineWidth', 2, 'MarkerSize', 7);
% hold on;

% plot( data(:,1), data(:,4), 'g-','LineWidth', 2, 'MarkerSize', 7);

% % legend('blue: best','fred: offline', 'green: online', 'northwest' );
% xlabel('epochs');
% ylabel('fitness');

% % legend(’black: best’,’fred: offline’, ’green: online’ );

% % figure ;
% % plot( data(:,1), data(:,2), 'k+','LineWidth', 2, 'MarkerSize', 7);
% % grid on;

% % figure ;
% % plot( data(:,1), data(:,3), 'k+','LineWidth', 2, 'MarkerSize', 7);
% % grid on;

% % figure ;
% % plot( data(:,1), data(:,4), 'k+','LineWidth', 2, 'MarkerSize', 7);
% % grid on;