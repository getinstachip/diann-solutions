module TopModule (
    input        clk,
    input        reset,
    input  [7:0] in,
    output [23:0] out_bytes,
    output       done
);

    // Instantiate FSM submodule
    FSM FSM_inst (
        .clk(clk),
        .reset(reset),
        .in(in),
        .done(done)
    );

    // Instantiate Datapath submodule
    Datapath Datapath_inst (
        .clk(clk),
        .reset(reset),
        .in(in),
        .out_bytes(out_bytes)
    );

endmodule