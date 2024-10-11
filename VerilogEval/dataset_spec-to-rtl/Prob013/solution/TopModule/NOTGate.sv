module NOTGate (
    input wire or_out,
    output wire out
);
    assign out = ~or_out;
endmodule