module TopModule (
    input wire clk,
    input wire areset,
    input wire bump_left,
    input wire bump_right,
    input wire ground,
    input wire dig,
    output wire walk_left,
    output wire walk_right,
    output wire aaah,
    output wire digging
);

    // Internal signals
    wire [2:0] state;
    wire fall_done;
    wire fall_enable;

    // Instantiate StateMachine
    StateMachine StateMachine_inst (
        .clk(clk),
        .areset(areset),
        .bump_left(bump_left),
        .bump_right(bump_right),
        .ground(ground),
        .dig(dig),
        .state(state)
    );

    // Define fall_enable based on the current state
    // Assuming state[2] indicates the falling state
    assign fall_enable = (state == 3'b100); // Adjust the state encoding as needed

    // Instantiate FallCounter
    FallCounter FallCounter_inst (
        .clk(clk),
        .enable(fall_enable),
        .reset(~fall_enable),
        .fall_done(fall_done)
    );

    // Instantiate OutputLogic
    OutputLogic OutputLogic_inst (
        .state(state),
        .fall_done(fall_done),
        .walk_left(walk_left),
        .walk_right(walk_right),
        .aaah(aaah),
        .digging(digging)
    );

endmodule