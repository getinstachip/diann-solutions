module TopModule (
    input  wire [255:0] in,
    input  wire [7:0]   sel,
    output wire         out
);

    // Instantiate the Mux256to1 submodule
    Mux256to1 u_Mux256to1 (
        .in(in),
        .sel(sel),
        .out(out)
    );

endmodule