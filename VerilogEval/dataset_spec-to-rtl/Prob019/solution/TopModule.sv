module TopModule (
    input in1,
    input in2,
    output out
);
    wire inverted_in2;

    // Instantiate InverterModule
    InverterModule u1 (
        .a(in2),
        .y(inverted_in2)
    );

    // Instantiate AndGateModule
    AndGateModule u2 (
        .a(in1),
        .b(inverted_in2),
        .y(out)
    );
endmodule