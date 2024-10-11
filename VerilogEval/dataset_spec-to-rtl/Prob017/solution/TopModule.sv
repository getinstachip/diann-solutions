module TopModule (
    input  wire [99:0] a,
    input  wire [99:0] b,
    input  wire        sel,
    output wire [99:0] out
);

    // Instantiate the Mux2to1 submodule
    Mux2to1 mux_inst (
        .a   (a),
        .b   (b),
        .sel (sel),
        .out (out)
    );

endmodule