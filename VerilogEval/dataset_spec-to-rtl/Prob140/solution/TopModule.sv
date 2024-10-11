module TopModule (
    input wire clk,
    input wire reset,
    input wire in,
    output wire disc,
    output wire flag,
    output wire err
);

    // Internal wire to connect StateMachine to OutputLogic
    wire [2:0] state;

    // Instantiate StateMachine
    StateMachine u_StateMachine (
        .clk(clk),
        .reset(reset),
        .in(in),
        .state(state)
    );

    // Instantiate OutputLogic
    OutputLogic u_OutputLogic (
        .state(state),
        .disc(disc),
        .flag(flag),
        .err(err)
    );

endmodule