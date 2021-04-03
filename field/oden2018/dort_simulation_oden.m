function [p, r] = dort_simulation_oden(nstreams, nlayer, layer_thickness, b, a, phase_type, g, n, depth, I_inc)

    cd('/Users/raphaellarouche/Documents/MATLAB/Dort2002_3/dort2002_v3')
    
    [~, sb] = size(b);
    [~, sa] = size(a);
    [~, sg] = size(g);
    [~, sn] = size(n);
    [~, slt] = size(layer_thickness);
    
    if isequal(sb, sa, sg, sn, slt, nlayer(1))
        
         % Settings 
         p = default_parameters_n();

         p.N = nstreams;  % The number of channel N will determined the Legendre polynomial degree (2*N - 1)
         p.N_layers = nlayer(1);
         p.sigma_s = b;
         p.sigma_a = a;
         p.phase_type = phase_type;
         %p.g = num2cell(g);
         p.g = g;
         p.n_refr = n;
         p.I_in = ones(p.N, 1) .* I_inc;
         
         % Angles
         p.phi_interpol = linspace(0, 2*pi, 73);
         
         % Depth 
         p.layer_thickness = layer_thickness;
         p.depth = depth';
         
         % Other settings 
         p.graphs_ABSDF = 0;
         p.graphs_BSDF = 0;
        
         % Simulation 
         r = dort2002(p, "forward", "angle resolved");
        
    else
        error("Size of arrays does not match number of layer.")
    end
    
end
