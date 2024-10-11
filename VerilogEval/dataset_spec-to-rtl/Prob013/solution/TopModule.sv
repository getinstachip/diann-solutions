module TopModule (
    input in1,
    input in2,
    output out
);
    wire or_out;

    ORGate u_ORGate (
        .in1(in1),
        .in2(in2),
        .or_out(or_out)
    );

    NOTGate u_NOTGate (
        .or_out(or_out),
        .out(out)
    );

endmodule