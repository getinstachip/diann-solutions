module TopModule (
    input  a,
    input  b,
    output out
);

    // Instantiate the NorGate submodule
    NorGate NorGate_inst (
        .a(a),
        .b(b),
        .out(out)
    );

endmodule