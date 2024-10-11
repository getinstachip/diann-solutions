module xor_gate_tb;

  // Declare signals
  logic a, b;
  logic y;

  // Instantiate the XOR gate
  XORGate dut (.a(a), .b(b), .y(y));

  // Stimulus generation
  initial begin
    $display("Starting XOR gate testbench");
    
    // Test all possible input combinations
    for (int i = 0; i < 4; i++) begin
      {a, b} = i;
      #10; // Wait for 10 time units
      
      // Check the output
      if (y !== (a ^ b)) begin
        $error("Test failed for inputs a=%b, b=%b. Expected y=%b, got y=%b", a, b, a ^ b, y);
      end else begin
        $display("Test passed for inputs a=%b, b=%b. Output y=%b", a, b, y);
      end
    end
    
    $display("XOR gate testbench completed");
    $finish;
  end

endmodule